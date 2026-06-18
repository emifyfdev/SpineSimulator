# -*- coding: utf-8 -*-
"""
SpineSimulator — Extensión para 3D Slicer
==========================================
Módulo scripted que integra el simulador de cirugía de columna vertebral
directamente en la interfaz de 3D Slicer.

Basado en SpineSimulator V4_1 (ex V3_60).

Uso:
  - Cargá los modelos STL/OBJ de las vértebras en Slicer
  - Abrí el módulo SpineSimulator desde el menú de módulos
  - Hacé click en "Iniciar simulador"
"""

import logging
import os
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import vtk
import qt
import numpy as np
import re
import copy
import tempfile
import time
import csv
from functools import partial


# ─── Constantes ───────────────────────────────────────────────────────────────

VERTEBRA_ORDER = (
    ["C1","C2","C3","C4","C5","C6","C7"] +
    ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"] +
    ["L1","L2","L3","L4","L5"] +
    ["S1","S2","S3","S4","S5","COXIS"]
)
ORDER_INDEX = {v: i for i, v in enumerate(VERTEBRA_ORDER)}

REGION_STIFFNESS = {
    "cervical": 0.50,
    "thoracic": 0.82,
    "lumbar":   0.40,
    "sacral":   1.00,
}

# Colores display (RGB 0-1)
COLOR_NORMAL    = (0.85, 0.75, 0.60)   # hueso
COLOR_SELECTED  = (0.30, 0.65, 0.90)   # azul claro
COLOR_ANCHOR    = (0.40, 0.80, 0.50)   # verde
COLOR_PIVOT     = (1.00, 0.90, 0.05)   # amarillo centro cuerpo
COLOR_DISC      = (0.10, 0.85, 1.00)   # celeste disco intervertebral

def get_region(label):
    u = label.upper()
    if re.match(r"^C[1-7]$", u): return "cervical"
    if u.startswith("T"): return "thoracic"
    if u.startswith("L"): return "lumbar"
    return "sacral"

def normalize_label(raw):
    """Reconoce columna completa en nombres de nodos de Slicer.

    Acepta C1-C7, T1-T12, L1-L5, S1-S5 y variantes frecuentes
    para coxis/coccyx/CO1. Devuelve una etiqueta canónica.
    """
    if raw is None:
        return None
    u = raw.upper()
    # Coxis/coccyx: se evalúa antes de C*, para evitar confusiones.
    if re.search(r'\b(COXIS|COXIX|COCCYX|COCCIX|COX|CO1|CX1)\b', u):
        return "COXIS"
    m = re.search(r'\b(C[1-7]|T(?:[1-9]|1[0-2])|D(?:[1-9]|1[0-2])|L[1-5]|S[1-5])\b', u)
    if not m:
        return None
    label = m.group(1)
    # En algunos modelos argentinos/españoles la dorsal se nombra D1-D12.
    if label.startswith("D"):
        label = "T" + label[1:]
    return label


# ─── FABRIK solver (igual que Fase 2, sin cambios) ───────────────────────────

class FABRIKSolver:
    def __init__(self, positions, stiffnesses, anchor_idx, iterations=15):
        self.pos       = [np.array(p, dtype=float) for p in positions]
        self.stiff     = list(stiffnesses)
        self.anchor_idx = anchor_idx
        self.iterations = iterations
        self.seg_lens  = [
            float(np.linalg.norm(self.pos[i+1] - self.pos[i]))
            for i in range(len(self.pos)-1)
        ]
        self.rest_pos  = [p.copy() for p in self.pos]

    def solve(self, target_idx, target_pos):
        n = len(self.pos)
        p = [v.copy() for v in self.pos]
        anchor = p[self.anchor_idx].copy()

        for _ in range(self.iterations):
            p[target_idx] = target_pos.copy()
            if target_idx > self.anchor_idx:
                for i in range(target_idx-1, self.anchor_idx-1, -1):
                    self._reach(p, i, i+1)
            else:
                for i in range(target_idx+1, self.anchor_idx+1):
                    self._reach(p, i, i-1)
            p[self.anchor_idx] = anchor.copy()
            if target_idx > self.anchor_idx:
                for i in range(self.anchor_idx+1, n):
                    self._reach(p, i, i-1)
            else:
                for i in range(self.anchor_idx-1, -1, -1):
                    self._reach(p, i, i+1)
            p[self.anchor_idx] = anchor.copy()

        result = []
        for i in range(n):
            if i == self.anchor_idx or i == target_idx:
                result.append(p[i])
            else:
                s = self.stiff[i]
                result.append(p[i]*(1.0-s) + self.rest_pos[i]*s)
        self.pos = result
        return result

    def _reach(self, p, i, j):
        seg_idx = min(i, j)
        if seg_idx >= len(self.seg_lens): return
        vec = p[i] - p[j]
        dist = np.linalg.norm(vec)
        if dist < 1e-6: return
        p[i] = p[j] + vec * (self.seg_lens[seg_idx] / dist)

    def reset(self):
        self.pos = [p.copy() for p in self.rest_pos]


# ─── Clase principal ──────────────────────────────────────────────────────────

class SpineSimulatorV3:

    def __init__(self):
        self.version        = "V3_60"
        self.scene          = slicer.mrmlScene
        self.model_nodes    = {}
        self.collision_model_nodes = {}
        self.original_model_nodes = {}
        self._original_display_state = {}
        self.transform_nodes = {}
        self.ordered_labels = []
        self.solver         = None
        self.anchor_label   = None
        self.active_label   = None
        # Por defecto la vértebra más caudal funciona como ancla fija.
        # Si el usuario activa el checkbox, también puede editar su movimiento
        # y ese movimiento se propaga a toda la cadena craneal.
        self.anchor_motion_enabled = False
        self._base_positions = {}
        self.vtp_output_dir = None
        self.converted_vtp_paths = {}
        # Rotaciones acumuladas por vértebra (rx, ry, rz en grados)
        self._rotations     = {}
        # Traslaciones acumuladas
        self._translations  = {}
        # Snapshot del estado previo
        self._prev_solver_pos   = None
        self._prev_transforms   = {}
        self._prev_matrices_flat = {}
        self._prev_rotations    = {}
        self._prev_translations = {}

        # Agrupa actualizaciones de matrices generadas por sliders/handle.
        self._dirty_transforms = False
        self._flush_timer = qt.QTimer()
        self._flush_timer.setInterval(20)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_transforms)


        # Dinámica distribuida: cuando se mueve una vértebra intermedia,
        # las vecinas también acompañan con un peso decreciente.
        # radius=1 afecta solo vecinas inmediatas; radius=2 afecta dos niveles, etc.
        self.dynamic_enabled = True
        self.influence_radius = 3
        self.influence_decay = 0.70
        self._panel         = None
        self._status_lbl    = None
        self._rot_widgets   = {}
        self._trans_widgets = {}
        self._anchor_combo = None
        self._anchor_label_widget = None
        self._selection_observer = None
        self._mouse_move_observer = None
        self._left_release_observer = None
        self._key_press_observer = None
        self._key_release_observer = None
        self._interactor_ref     = None

        # Interacción nativa de transformada de Slicer.
        # No se crean aros/modelos extra: se habilitan los handles nativos
        # del vtkMRMLLinearTransformNode de la vértebra seleccionada.
        self.native_transform_interaction_enabled = True
        self.native_rotation_only = False
        self.native_handle_scale = 1.0
        # Offset visual del handle: el gizmo se puede mostrar desplazado hacia
        # la derecha de la pantalla para no tapar la anatomía. IMPORTANTE:
        # el movimiento matemático sigue usando el disco/fiducial real como pivot.
        self.native_handle_screen_offset_enabled = True
        self.native_handle_screen_offset_mm = 65.0
        self._native_interaction_display_node = None
        # Handle nativo único de Slicer: NO es un modelo/gizmo propio.
        # Es un vtkMRMLLinearTransformNode visible/interactivo, colocado exactamente
        # en el fiducial de disco/pivot activo. El modelo se sigue moviendo con la
        # matriz anatómica propia del simulador.
        self._interaction_handle_node = None
        self._interaction_handle_observer = None
        self._interaction_handle_updating = False
        self._interaction_handle_active_label = None
        # No se crean modelos/gizmos propios. Se usa la interacción nativa de transformadas de Slicer.

        # Fiduciales amarillos para visualizar el centro/pivot real de cada vértebra.
        self.pivot_fiducial_node = None
        self._pivot_point_indices = {}
        self.show_pivot_fiducials = False

        # Fiduciales celestes que representan discos intervertebrales.
        # Cada disco se calcula entre dos centros de cuerpo vertebral: Disc_C5_C6, etc.
        # Si está activo, estos discos son el pivot de movimiento del segmento craneal.
        self.disc_fiducial_node = None
        self._disc_point_indices = {}  # label craneal -> índice del punto Disc_label_caudal
        self.show_disc_fiducials = False
        self.use_disc_pivots = True
        self.live_disc_pivots = True
        self._motion_pivots = {}       # pivots locales usados para movimiento; por defecto discos

        # Modo de pivot:
        # - BODY_DENSITY_POS_Y: intenta ubicar el centro funcional en el cuerpo vertebral,
        #   descartando apófisis por recorte anterior + densidad de puntos. Asume anterior = +Y RAS.
        # - BODY_DENSITY_NEG_Y: lo mismo, pero asume anterior = -Y.
        # - BOUNDS_CENTER: centro del bound completo de la malla.
        # - CENTER_OF_MASS: centro de masa de todos los puntos de la malla.
        # - MANUAL_FIDUCIALS: usa los fiduciales Pivot_XX movidos por el usuario.
        self.pivot_mode = "BODY_DENSITY_POS_Y"
        self.manual_pivots_enabled = False
        # Si está activo, el movimiento lee la posición ACTUAL de los fiduciales Pivot_XX
        # antes de cada rotación/traslación. Esto evita usar una copia vieja del pivot.
        self.live_fiducial_pivots = False

        # Cadena cinemática: la rotación de una vértebra caudal arrastra a las
        # vértebras craneales como si fueran hijas de un hueso en Pose Mode.
        # Ejemplo: si rota C6, C5-C4-C3 se proyectan hacia adelante por el arco
        # de C6, en vez de girar cada una sobre su propio eje.
        self.kinematic_chain_enabled = True
        # Local bend agrega una pequeña rotación propia a las vértebras craneales
        # para suavizar la curva. 0.0 = solo arrastre rígido por padre.
        self.local_bend_fraction = 0.60
        self.debug_enabled = False
        self._event_log = []
        # Colisiones V29: se chequean sobre las copias VTP transformadas.
        # Primero se usan bounds para descartar pares lejanos y solo después
        # distancia superficie-superficie para pintar contactos; el bloqueo es opcional.
        self.collision_enabled = True
        self.collision_blocking_enabled = True
        self.collision_margin_mm = 0.5
        self.collision_neighbor_radius = 1
        self.collision_max_sample_points = 600
        self.collision_proxy_enabled = True
        self.collision_proxy_reduction = 0.85
        self.collision_heatmap_max_points = 900
        self.collision_heatmap_enabled = True
        self.collision_heatmap_mode = "PATCH"  # PATCH, SPHERES o SURFACE
        self.collision_heatmap_radius_mm = 3.0
        self._collision_baseline_pairs = {}
        self._last_collision = None
        self._collision_color_node = None
        self._collision_heat_labels = set()
        self._collision_overlay_nodes = []
        self.contact_marks_persistent = True
        self.contact_marks_follow_vertebra = True
        self._contact_mark_nodes_by_pair = {}
        self._contact_heat_centers_by_label = {}
        self.osteotomy_enabled = True
        self.osteotomy_mouse_mode = False
        self.osteotomy_continuous_drill = True
        self._osteotomy_drilling = False
        self._osteotomy_dirty_labels = set()
        self._osteotomy_exact_collision_labels = set()
        self._osteotomy_stroke_label = None
        self._osteotomy_stroke_centers = []
        self._last_drill_time = 0.0
        self._last_drill_center_local = None
        self.osteotomy_drill_interval_sec = 0.08
        self.osteotomy_drill_min_step_mm = 0.8
        self.osteotomy_radius_mm = 3.0
        self._osteotomy_preview_node = None
        self._osteotomy_preview_label = None
        self._osteotomy_center_local = {}
        self._osteotomy_original_polydata = {}
        self._osteotomy_cut_count = {}
        self._sh_root_folder_id = None
        self._sh_folder_ids = {}

    # ── Organización visual en Subject Hierarchy ─────────────────────────────

    def _subject_hierarchy_node(self):
        try:
            return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(self.scene)
        except Exception:
            return None

    def _ensure_scene_folders(self):
        """Crea una carpeta visual para todos los nodos generados por el simulador."""
        sh = self._subject_hierarchy_node()
        if not sh:
            return False
        try:
            if self._sh_root_folder_id and sh.GetItemName(self._sh_root_folder_id):
                return True
        except Exception:
            pass

        try:
            scene_item = sh.GetSceneItemID()
            self._sh_root_folder_id = sh.CreateFolderItem(scene_item, f"SpineSimulator {self.version}")
            folder_names = {
                "vtp": "01 VTP visuales",
                "proxies": "02 Proxies de colisión",
                "transforms": "03 Transforms",
                "fiducials": "04 Fiduciales pivots/discos",
                "contacts": "05 Marcas de contacto",
                "colors": "06 Tablas de color",
                "osteotomy": "07 Osteotomía virtual",
            }
            self._sh_folder_ids = {}
            for key, name in folder_names.items():
                self._sh_folder_ids[key] = sh.CreateFolderItem(self._sh_root_folder_id, name)
            return True
        except Exception as e:
            self._log_event("WARN", f"No se pudo crear carpeta de escena: {e}")
            return False

    def _add_node_to_scene_folder(self, node, folder_key):
        """Mueve un nodo generado a la subcarpeta visual correspondiente."""
        if node is None or not self._ensure_scene_folders():
            return
        sh = self._subject_hierarchy_node()
        if not sh:
            return
        folder_id = self._sh_folder_ids.get(folder_key) or self._sh_root_folder_id
        try:
            item_id = sh.GetItemByDataNode(node)
            if item_id:
                sh.SetItemParent(item_id, folder_id)
        except Exception:
            pass

    def _cleanup_scene_folders(self):
        """Elimina la carpeta visual del simulador si queda vacía al detener."""
        sh = self._subject_hierarchy_node()
        if not sh:
            self._sh_root_folder_id = None
            self._sh_folder_ids = {}
            return
        try:
            if self._sh_root_folder_id:
                sh.RemoveItem(self._sh_root_folder_id)
        except Exception:
            pass
        self._sh_root_folder_id = None
        self._sh_folder_ids = {}

    # ── API pública ──────────────────────────────────────────────────────────

    def start(self):
        if not self._detect_and_build():
            return
        self._build_panel()
        self._install_click_observer()
        self._paint_all_normal()
        self._update_native_transform_interaction()
        print(f"\n[SpineSimulator {self.version}] Listo.")
        print("  → Click en vértebra 3D para seleccionar")
        print("  → Rotación como movimiento principal, traslación disponible")
        print("  → V39: drill continuo manteniendo click y arrastrando")

    def stop(self):
        self._remove_click_observer()
        self._disable_all_native_transform_interactions()
        self._remove_osteotomy_preview()
        self._clear_collision_heatmap()
        if self._collision_color_node:
            try:
                self.scene.RemoveNode(self._collision_color_node)
            except Exception:
                pass
            self._collision_color_node = None
        self._cleanup_rotation_gizmo()
        self._cleanup_pivot_fiducials()
        self._cleanup_disc_fiducials()
        self._cleanup_transforms()
        self._cleanup_converted_vtp_nodes()
        self._restore_original_model_visibility()
        self._cleanup_scene_folders()
        try:
            self._flush_timer.stop()
            self._dirty_transforms = False
        except Exception:
            pass
        if self._panel:
            self._panel.close()
            self._panel = None
        self.model_nodes = {}
        self.collision_model_nodes = {}
        self.transform_nodes = {}
        self.active_label = None
        self._osteotomy_center_local = {}
        self._osteotomy_original_polydata = {}
        self._osteotomy_cut_count = {}
        print(f"[SpineSimulator {self.version}] Detenido. Escena restaurada.")

    # ── Detección y construcción ─────────────────────────────────────────────

    def _detect_and_build(self):
        found = {}

        # Evita que una corrida anterior vuelva a detectar las copias VTP como entrada.
        self._cleanup_converted_vtp_nodes()
        self._ensure_scene_folders()

        for i in range(self.scene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = self.scene.GetNthNodeByClass(i, "vtkMRMLModelNode")
            name = node.GetName() or ""
            if name.startswith("SpineVTP_") or name.startswith("SpineRotGizmo_") or name.startswith("SpineCollisionProxy_") or (name.startswith("SpineContactOverlay_") or name.startswith("SpineContactPatch_")):
                continue
            label = normalize_label(name)
            if label and label in ORDER_INDEX and label not in found:
                found[label] = node
        if not found:
            print("[ERROR] No se encontraron modelos vertebrales.")
            return False

        self.ordered_labels = sorted(found.keys(), key=lambda l: ORDER_INDEX[l])
        self.original_model_nodes = {l: found[l] for l in self.ordered_labels}
        self.anchor_label   = self.ordered_labels[-1]

        print(f"\n[SpineSimulator {self.version}] {len(self.ordered_labels)} vértebras: "
              + " → ".join(self.ordered_labels))
        print(f"  Ancla inicial: {self.anchor_label}")

        # Conversión previa: antes de crear transforms, solver y panel.
        # Desde este punto el simulador usa las copias VTP, no los nodos STL/originales.
        self.model_nodes = self._convert_detected_models_to_vtp(self.original_model_nodes)
        if not self.model_nodes:
            print("[ERROR] Falló la conversión a VTP.")
            return False
        self._create_collision_proxy_nodes()

        self._cleanup_transforms()
        self._compute_base_positions()
        self._create_transforms()
        self._create_pivot_fiducials()
        self._create_disc_fiducials()
        self._read_disc_fiducials_as_motion_pivots(update_status=False)
        self._build_solver()

        for l in self.ordered_labels:
            self._rotations[l]    = [0.0, 0.0, 0.0]
            self._translations[l] = [0.0, 0.0, 0.0]

        self._make_models_pickable()
        self._calibrate_collision_baseline()
        return True

    def _remember_original_display_state(self, node):
        """Guarda visibilidad/seleccionabilidad para restaurar la escena al cerrar."""
        if not node:
            return
        dn = node.GetDisplayNode()
        if not dn:
            return
        try:
            self._original_display_state[node.GetID()] = {
                "visibility": dn.GetVisibility(),
                "opacity": dn.GetOpacity(),
                "color": tuple(dn.GetColor()),
                "selectable": node.GetSelectable() if hasattr(node, "GetSelectable") else None,
            }
        except Exception:
            pass

    def _restore_original_model_visibility(self):
        """Restaura los STL/modelos originales ocultados durante la conversión VTP."""
        for node in list(self.original_model_nodes.values()):
            if not node:
                continue
            dn = node.GetDisplayNode()
            state = self._original_display_state.get(node.GetID(), {})
            if dn:
                try:
                    dn.SetVisibility(int(state.get("visibility", 1)))
                except Exception:
                    dn.SetVisibility(True)
                try:
                    dn.SetOpacity(float(state.get("opacity", dn.GetOpacity())))
                except Exception:
                    pass
                try:
                    color = state.get("color")
                    if color:
                        dn.SetColor(*color)
                except Exception:
                    pass
            selectable = state.get("selectable")
            if selectable is not None:
                try:
                    node.SetSelectable(int(selectable))
                except Exception:
                    pass

    def _cleanup_converted_vtp_nodes(self):
        """Elimina copias VTP generadas por corridas anteriores del simulador."""
        to_remove = []
        for i in range(self.scene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = self.scene.GetNthNodeByClass(i, "vtkMRMLModelNode")
            nname = node.GetName() or ""
            if nname.startswith("SpineVTP_") or nname.startswith("SpineCollisionProxy_") or nname.startswith("SpineContactOverlay_") or nname.startswith("SpineContactPatch_"):
                to_remove.append(node)
        for node in to_remove:
            try:
                self.scene.RemoveNode(node)
            except Exception:
                pass
        self.converted_vtp_paths = {}
        self.collision_model_nodes = {}
        self._collision_overlay_nodes = []
        self._contact_mark_nodes_by_pair = {}

    def _get_vtp_output_dir(self):
        """Carpeta automática para guardar los .vtp convertidos."""
        try:
            base = slicer.app.temporaryPath
        except Exception:
            base = tempfile.gettempdir()
        folder = os.path.join(base, "SpineSimulator_VTP")
        os.makedirs(folder, exist_ok=True)
        self.vtp_output_dir = folder
        return folder

    def _prepare_polydata_for_vtp(self, polydata):
        """
        Prepara la malla antes de escribirla como .vtp:
        - triangula, por seguridad
        - limpia puntos duplicados
        - recalcula normales consistentes

        No decima la malla: esto mantiene la geometría original. La optimización de
        módulos extra pueden agregarse después con mallas secundarias.
        """
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(polydata)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(tri.GetOutputPort())

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(normals.GetOutput())
        return out

    def _make_collision_proxy_polydata(self, polydata):
        """Crea una malla liviana para contacto en vivo, sin tocar la malla visual."""
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        source = vtk.vtkPolyData()
        source.DeepCopy(polydata)

        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(source)

        dec = vtk.vtkDecimatePro()
        dec.SetInputConnection(tri.GetOutputPort())
        dec.SetTargetReduction(max(0.0, min(0.95, float(self.collision_proxy_reduction))))
        dec.PreserveTopologyOn()
        dec.BoundaryVertexDeletionOff()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(dec.GetOutputPort())

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(normals.GetOutput())
        if out.GetNumberOfPoints() < 50:
            out.DeepCopy(source)
        return out

    def _create_collision_proxy_nodes(self):
        """Crea proxies VTP invisibles para acelerar detección de contacto."""
        self.collision_model_nodes = {}
        if not self.collision_proxy_enabled:
            return
        for label in self.ordered_labels:
            visual = self.model_nodes.get(label)
            if not visual or not visual.GetPolyData():
                continue
            proxy_poly = self._make_collision_proxy_polydata(visual.GetPolyData())
            if proxy_poly is None:
                continue
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
            node.SetName(f"SpineCollisionProxy_{label}")
            node.SetAndObservePolyData(proxy_poly)
            node.CreateDefaultDisplayNodes()
            self._add_node_to_scene_folder(node, "proxies")
            dn = node.GetDisplayNode()
            if dn:
                dn.SetVisibility(False)
                self._safe_call(dn, "SetVisibility3D", False)
                self._safe_call(dn, "SetPickable", False)
            try:
                node.SetSelectable(0)
            except Exception:
                pass
            self.collision_model_nodes[label] = node
            print(f"  Proxy colisión {label}: {visual.GetPolyData().GetNumberOfPoints()} pts → {proxy_poly.GetNumberOfPoints()} pts")

    def _write_vtp(self, polydata, filepath):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(polydata)
        writer.SetDataModeToBinary()
        ok = writer.Write()
        return bool(ok)

    def _convert_detected_models_to_vtp(self, original_nodes):
        """
        Convierte cada modelo detectado a PolyData VTP antes de abrir la interfaz.

        Flujo:
            modelo original/STL en escena
            → vtkTriangleFilter / vtkCleanPolyData / vtkPolyDataNormals
            → archivo .vtp en carpeta temporal
            → nuevo vtkMRMLModelNode llamado SpineVTP_<label>
            → se oculta el original
            → el simulador usa el nodo SpineVTP_<label>
        """
        folder = self._get_vtp_output_dir()
        converted = {}
        self.converted_vtp_paths = {}

        print(f"  Convirtiendo modelos a VTP en: {folder}")

        for label in self.ordered_labels:
            src = original_nodes[label]
            self._remember_original_display_state(src)
            poly = src.GetPolyData()
            if poly is None or poly.GetNumberOfPoints() == 0:
                print(f"  [ERROR] {label}: modelo sin PolyData válido.")
                continue

            vtp_poly = self._prepare_polydata_for_vtp(poly)
            path = os.path.join(folder, f"SpineVTP_{label}.vtp")
            if not self._write_vtp(vtp_poly, path):
                print(f"  [ERROR] {label}: no se pudo escribir {path}")
                continue

            # Incrustar el label en el FieldData del polydata para que el picker
            # pueda identificar la vértebra sin depender de comparaciones de
            # punteros VTK (que cambian entre versiones de Slicer/VTK).
            _arr = vtk.vtkStringArray()
            _arr.SetName("SpineLabel")
            _arr.InsertNextValue(label)
            vtp_poly.GetFieldData().AddArray(_arr)

            new_node = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
            new_node.SetName(f"SpineVTP_{label}")
            new_node.SetAndObservePolyData(vtp_poly)
            new_node.CreateDefaultDisplayNodes()
            self._add_node_to_scene_folder(new_node, "vtp")

            # Copiar propiedades visuales básicas del original.
            src_dn = src.GetDisplayNode()
            new_dn = new_node.GetDisplayNode()
            if src_dn and new_dn:
                new_dn.SetColor(src_dn.GetColor())
                new_dn.SetOpacity(src_dn.GetOpacity())
                new_dn.SetVisibility(True)
                src_dn.SetVisibility(False)

            converted[label] = new_node
            self.converted_vtp_paths[label] = path
            print(f"  {label}: {poly.GetNumberOfPoints()} pts / {poly.GetNumberOfCells()} celdas"
                  f" → VTP {vtp_poly.GetNumberOfPoints()} pts / {vtp_poly.GetNumberOfCells()} celdas")

        missing = [l for l in self.ordered_labels if l not in converted]
        if missing:
            print("[ERROR] No se pudieron convertir: " + ", ".join(missing))
            return {}

        print("  Conversión VTP terminada. Se abre interfaz del simulador.\n")
        return converted

    def _polydata_points_numpy(self, poly):
        """Devuelve puntos del PolyData como ndarray Nx3. Fallback sin numpy_support."""
        if poly is None or poly.GetPoints() is None:
            return np.zeros((0, 3), dtype=float)
        pts_vtk = poly.GetPoints().GetData()
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            pts = vtk_to_numpy(pts_vtk).astype(float, copy=False)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                return pts[:, :3]
        except Exception:
            pass
        n = poly.GetNumberOfPoints()
        pts = np.zeros((n, 3), dtype=float)
        for i in range(n):
            pts[i] = poly.GetPoint(i)
        return pts

    def _bounds_center_from_polydata(self, poly):
        b = [0.0] * 6
        poly.GetBounds(b)
        return np.array([
            (b[0] + b[1]) * 0.5,
            (b[2] + b[3]) * 0.5,
            (b[4] + b[5]) * 0.5,
        ], dtype=float)

    def _center_of_mass_from_points(self, pts, fallback):
        if pts is None or len(pts) < 3:
            return np.array(fallback, dtype=float)
        return np.mean(pts, axis=0).astype(float)

    def _body_density_pivot_from_polydata(self, poly, anterior_sign=+1):
        """
        Estima el pivot en el CUERPO VERTEBRAL, no en toda la vértebra.

        Idea:
        1) Leer la nube de puntos de la superficie.
        2) Recortar la zona central izquierda/derecha y superior/inferior para sacar
           apófisis transversas y extremos muy periféricos.
        3) Quedarse con la mitad/porción anterior en eje Y RAS para alejar la espinosa.
           En Slicer/RAS normalmente anterior = +Y. Si está invertido, usar modo -Y.
        4) Voxelizar los puntos restantes y calcular el promedio ponderado de los voxeles
           más densos. Eso tiende a caer en el bloque compacto del cuerpo vertebral.

        No es una segmentación anatómica perfecta, pero suele ser mejor que el centro
        geométrico de toda la malla cuando la apófisis espinosa tira el centro hacia posterior.
        """
        fallback = self._bounds_center_from_polydata(poly)
        pts = self._polydata_points_numpy(poly)
        if pts.shape[0] < 50:
            return fallback, {"method": "fallback_bounds", "points": int(pts.shape[0])}

        # Percentiles robustos para ignorar outliers o puntas largas.
        x0, x1 = np.percentile(pts[:, 0], [15, 85])
        z0, z1 = np.percentile(pts[:, 2], [10, 90])

        # En RAS: +Y suele ser anterior. Si la orientación está invertida,
        # cambiar a BODY_DENSITY_NEG_Y desde la interfaz.
        if anterior_sign >= 0:
            y_cut = np.percentile(pts[:, 1], 42)
            mask_y = pts[:, 1] >= y_cut
        else:
            y_cut = np.percentile(pts[:, 1], 58)
            mask_y = pts[:, 1] <= y_cut

        mask = (
            (pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
            (pts[:, 2] >= z0) & (pts[:, 2] <= z1) &
            mask_y
        )
        cand = pts[mask]

        # Si el recorte fue excesivo, relajar a solo anterior/posterior + zona central LR.
        if cand.shape[0] < max(80, int(0.03 * pts.shape[0])):
            x0, x1 = np.percentile(pts[:, 0], [10, 90])
            mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & mask_y
            cand = pts[mask]

        if cand.shape[0] < 30:
            return fallback, {"method": "fallback_bounds_after_crop", "points": int(cand.shape[0])}

        # Voxelización para encontrar la región de mayor concentración de puntos.
        mins = cand.min(axis=0)
        maxs = cand.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        voxel = max(diag / 18.0, 0.5)  # mm aproximados; evita voxeles absurdamente pequeños
        ijk = np.floor((cand - mins) / voxel).astype(np.int64)

        uniq, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
        if len(counts) == 0:
            return self._center_of_mass_from_points(cand, fallback), {"method": "crop_mean", "points": int(cand.shape[0])}

        # Tomar los voxeles más densos, no solo el máximo, para evitar que un parche de STL
        # localmente sobredensificado mande el pivot a una pared de la malla.
        threshold = np.percentile(counts, 82)
        dense_voxels = np.where(counts >= threshold)[0]
        if dense_voxels.size < 1:
            dense_voxels = np.array([int(np.argmax(counts))])
        dense_mask = np.isin(inv, dense_voxels)
        dense_pts = cand[dense_mask]

        if dense_pts.shape[0] < 20:
            pivot = self._center_of_mass_from_points(cand, fallback)
            method = "crop_mean_low_density"
        else:
            # Promedio de los puntos en los voxeles más densos.
            pivot = np.mean(dense_pts, axis=0).astype(float)
            method = "body_density_voxels"

        info = {
            "method": method,
            "points_total": int(pts.shape[0]),
            "points_crop": int(cand.shape[0]),
            "points_dense": int(dense_pts.shape[0]),
            "voxel_mm": float(voxel),
            "anterior_sign": int(1 if anterior_sign >= 0 else -1),
        }
        return pivot, info

    def _pivot_from_polydata_by_mode(self, label, poly):
        """Calcula el pivot local según el modo elegido."""
        mode = getattr(self, "pivot_mode", "BODY_DENSITY_POS_Y")
        if mode == "BOUNDS_CENTER":
            return self._bounds_center_from_polydata(poly), {"method": "bounds_center"}
        if mode == "CENTER_OF_MASS":
            pts = self._polydata_points_numpy(poly)
            return self._center_of_mass_from_points(pts, self._bounds_center_from_polydata(poly)), {"method": "all_points_mean", "points": int(len(pts))}
        if mode == "BODY_DENSITY_NEG_Y":
            return self._body_density_pivot_from_polydata(poly, anterior_sign=-1)
        # Default: cuerpo vertebral por densidad, anterior +Y RAS.
        return self._body_density_pivot_from_polydata(poly, anterior_sign=+1)

    def _compute_base_positions(self):
        """
        Calcula pivotes locales. Por defecto ya NO usa el centro de toda la malla,
        sino una estimación automática del centro funcional del CUERPO VERTEBRAL
        por recorte anatómico aproximado + densidad de puntos.
        """
        self._base_positions = {}
        for label in self.ordered_labels:
            poly = self.model_nodes[label].GetPolyData()
            if poly is None:
                continue
            pivot, info = self._pivot_from_polydata_by_mode(label, poly)
            self._base_positions[label] = np.array(pivot, dtype=float)
            method = info.get("method", "?") if isinstance(info, dict) else "?"
            print(f"  Pivot {label}: {method} = "
                  f"({self._base_positions[label][0]:.3f}, "
                  f"{self._base_positions[label][1]:.3f}, "
                  f"{self._base_positions[label][2]:.3f})")

    def _create_transforms(self):
        for label in self.ordered_labels:
            t = self.scene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
            t.SetName(f"SpineV3Center_{label}")
            self.transform_nodes[label] = t
            self._add_node_to_scene_folder(t, "transforms")
            self.model_nodes[label].SetAndObserveTransformNodeID(t.GetID())
            if label in self.collision_model_nodes:
                self.collision_model_nodes[label].SetAndObserveTransformNodeID(t.GetID())

    def _build_solver(self):
        positions    = [self._base_positions[l].tolist() for l in self.ordered_labels]
        stiffnesses  = [REGION_STIFFNESS[get_region(l)] for l in self.ordered_labels]
        if self.anchor_label in self.ordered_labels:
            anchor_idx = self.ordered_labels.index(self.anchor_label)
        else:
            anchor_idx = len(self.ordered_labels) - 1
            self.anchor_label = self.ordered_labels[anchor_idx]
        self.solver  = FABRIKSolver(positions, stiffnesses, anchor_idx)

    def _cleanup_transforms(self):
        to_remove = []
        for i in range(self.scene.GetNumberOfNodesByClass("vtkMRMLLinearTransformNode")):
            node = self.scene.GetNthNodeByClass(i, "vtkMRMLLinearTransformNode")
            nname = node.GetName() or ""
            if nname.startswith("SpineV3Center_") or nname.startswith("SpineV3Handle_"):
                to_remove.append(node)
        for node in to_remove:
            if node == self._interaction_handle_node and self._interaction_handle_observer:
                try:
                    node.RemoveObserver(self._interaction_handle_observer)
                except Exception:
                    pass
                self._interaction_handle_observer = None
                self._interaction_handle_node = None
            self.scene.RemoveNode(node)
        self.transform_nodes = {}

    # ── Fiduciales amarillos de centro/pivot ─────────────────────────────────

    def _cleanup_pivot_fiducials(self):
        """Elimina fiduciales de pivote creados por corridas anteriores."""
        to_remove = []
        for i in range(self.scene.GetNumberOfNodesByClass("vtkMRMLMarkupsFiducialNode")):
            node = self.scene.GetNthNodeByClass(i, "vtkMRMLMarkupsFiducialNode")
            if (node.GetName() or "").startswith("SpineV3_Pivots_Cuerpo"):
                to_remove.append(node)
        for node in to_remove:
            try:
                self.scene.RemoveNode(node)
            except Exception:
                pass
        self.pivot_fiducial_node = None
        self._pivot_point_indices = {}

    def _create_pivot_fiducials(self):
        """Crea un punto amarillo por vértebra en el centro usado como pivot de rotación."""
        self._cleanup_pivot_fiducials()
        node = self.scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SpineV3_Pivots_Cuerpo")
        self.pivot_fiducial_node = node
        self._pivot_point_indices = {}
        self._add_node_to_scene_folder(node, "fiducials")

        try:
            node.CreateDefaultDisplayNodes()
        except Exception:
            pass

        dn = node.GetDisplayNode()
        if dn:
            try:
                dn.SetColor(*COLOR_PIVOT)
            except Exception:
                pass
            try:
                dn.SetSelectedColor(*COLOR_PIVOT)
            except Exception:
                pass
            try:
                dn.SetGlyphTypeFromString("Sphere3D")
            except Exception:
                pass
            try:
                dn.SetGlyphScale(3.0)
            except Exception:
                pass
            try:
                dn.SetTextScale(2.0)
            except Exception:
                pass
            try:
                dn.SetPointLabelsVisibility(True)
            except Exception:
                pass
            try:
                dn.SetVisibility(bool(self.show_pivot_fiducials))
            except Exception:
                pass

        for label in self.ordered_labels:
            pos = self._get_current_pivot_world(label)
            point_name = f"Pivot_{label}"
            idx = self._add_fiducial_point(node, pos, point_name)
            self._pivot_point_indices[label] = idx

        print(f"[Pivots] {len(self._pivot_point_indices)} fiduciales amarillos creados en los pivotes estimados del cuerpo vertebral.")

    def _add_fiducial_point(self, node, pos, name):
        """Compatibilidad entre versiones de Slicer para agregar un fiducial."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        try:
            idx = node.AddControlPoint(vtk.vtkVector3d(x, y, z), name)
        except Exception:
            try:
                idx = node.AddFiducial(x, y, z)
                node.SetNthControlPointLabel(idx, name)
            except Exception:
                idx = node.GetNumberOfControlPoints()
                node.AddControlPointWorld(vtk.vtkVector3d(x, y, z), name)
        try:
            node.SetNthControlPointLocked(idx, False)
        except Exception:
            pass
        return idx

    def _get_current_pivot_world(self, label):
        """Devuelve el centro/pivot de la malla en coordenadas RAS/world actuales."""
        pivot = self._base_positions.get(label)
        if pivot is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        if label not in self.transform_nodes:
            return np.array(pivot, dtype=float)

        m = vtk.vtkMatrix4x4()
        self.transform_nodes[label].GetMatrixTransformToWorld(m)
        p = [float(pivot[0]), float(pivot[1]), float(pivot[2]), 1.0]
        out = [0.0, 0.0, 0.0, 0.0]
        m.MultiplyPoint(p, out)
        return np.array(out[:3], dtype=float)

    def _update_pivot_fiducials(self):
        """Actualiza fiduciales SOLO cuando los pivotes son automáticos.

        Importante: si el usuario movió Pivot_XX al cuerpo vertebral, NO hay que
        reescribir esos puntos en cada render. Ese era el motivo por el que parecía
        que el programa no tomaba los fiduciales corregidos: los volvía a pisar con
        el pivot automático anterior.
        """
        if self.manual_pivots_enabled or self.live_fiducial_pivots or self.pivot_mode == "MANUAL_FIDUCIALS":
            return
        node = self.pivot_fiducial_node
        if not node:
            return
        for label, idx in self._pivot_point_indices.items():
            pos = self._get_current_pivot_world(label)
            try:
                node.SetNthControlPointPositionWorld(idx, float(pos[0]), float(pos[1]), float(pos[2]))
            except Exception:
                try:
                    node.SetNthControlPointPosition(idx, float(pos[0]), float(pos[1]), float(pos[2]))
                except Exception:
                    pass

    def _read_fiducial_position_world(self, idx):
        """Lee un control point en coordenadas RAS/world, compatible entre versiones."""
        node = self.pivot_fiducial_node
        if not node:
            return None
        pos = [0.0, 0.0, 0.0]
        try:
            node.GetNthControlPointPositionWorld(idx, pos)
            return np.array(pos, dtype=float)
        except Exception:
            try:
                node.GetNthControlPointPosition(idx, pos)
                return np.array(pos, dtype=float)
            except Exception:
                return None

    def _refresh_base_positions_from_fiducials(self, rebuild_solver=False, update_status=False):
        """Actualiza self._base_positions usando la posición ACTUAL de Pivot_XX.

        Se llama antes de cada movimiento en modo manual/live, para que si el usuario
        arrastra un fiducial al centro del cuerpo vertebral, el próximo slider use
        inmediatamente ese punto como centro de rotación.
        """
        node = self.pivot_fiducial_node
        if not node:
            return 0
        updated = 0
        for label, idx in self._pivot_point_indices.items():
            pos = self._read_fiducial_position_world(idx)
            if pos is None:
                continue
            world = [float(pos[0]), float(pos[1]), float(pos[2]), 1.0]
            if label in self.transform_nodes:
                m = vtk.vtkMatrix4x4()
                inv = vtk.vtkMatrix4x4()
                self.transform_nodes[label].GetMatrixTransformToWorld(m)
                vtk.vtkMatrix4x4.Invert(m, inv)
                out = [0.0, 0.0, 0.0, 0.0]
                inv.MultiplyPoint(world, out)
                self._base_positions[label] = np.array(out[:3], dtype=float)
            else:
                self._base_positions[label] = np.array(world[:3], dtype=float)
            updated += 1
        if updated:
            self.manual_pivots_enabled = True
            self.live_fiducial_pivots = True
            self.pivot_mode = "MANUAL_FIDUCIALS"
            if rebuild_solver:
                self._build_solver()
            if update_status:
                self._update_status(f"Pivotes leídos desde fiduciales actuales: {updated}.")
        return updated

    def _use_current_fiducials_as_pivots(self):
        """Activa el modo manual/live y usa los fiduciales Pivot_XX como pivotes reales."""
        if not self.pivot_fiducial_node:
            self._update_status("No hay fiduciales de pivot para leer.")
            return
        updated = self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)
        self.manual_pivots_enabled = True
        self.live_fiducial_pivots = True
        self.pivot_mode = "MANUAL_FIDUCIALS"
        # Si estamos usando discos, recalculamos Disc_XX_YY a partir de los nuevos centros del cuerpo.
        if self.use_disc_pivots:
            self._recalculate_disc_fiducials_from_body_centers()
        self._apply_all_transforms()
        self._update_status(f"Modo manual LIVE: usando centros Pivot_XX actuales ({updated}); discos actualizados si están activos.")

    def _recompute_automatic_pivots(self):
        self.manual_pivots_enabled = False
        self.live_fiducial_pivots = False
        if self.pivot_mode == "MANUAL_FIDUCIALS":
            self.pivot_mode = "BODY_DENSITY_POS_Y"
        self._compute_base_positions()
        self._build_solver()
        self._apply_all_transforms()
        self._update_pivot_fiducials()
        self._update_native_transform_interaction()
        if self.disc_fiducial_node:
            self._recalculate_disc_fiducials_from_body_centers()

    def _set_pivot_fiducials_visible(self, visible):
        self.show_pivot_fiducials = bool(visible)
        if self.pivot_fiducial_node and self.pivot_fiducial_node.GetDisplayNode():
            self.pivot_fiducial_node.GetDisplayNode().SetVisibility(bool(visible))

    # ── Fiduciales celestes de discos intervertebrales ───────────────────────

    def _cleanup_disc_fiducials(self):
        """Elimina fiduciales Disc_XX_YY creados por corridas anteriores."""
        to_remove = []
        for i in range(self.scene.GetNumberOfNodesByClass("vtkMRMLMarkupsFiducialNode")):
            node = self.scene.GetNthNodeByClass(i, "vtkMRMLMarkupsFiducialNode")
            if (node.GetName() or "").startswith("SpineV3_Discos"):
                to_remove.append(node)
        for node in to_remove:
            try:
                self.scene.RemoveNode(node)
            except Exception:
                pass
        self.disc_fiducial_node = None
        self._disc_point_indices = {}
        self._motion_pivots = {}

    def _caudal_neighbor_label(self, label):
        """Devuelve la vértebra caudal inmediata. ordered_labels es craneal→caudal."""
        try:
            idx = self.ordered_labels.index(label)
        except ValueError:
            return None
        if idx >= len(self.ordered_labels) - 1:
            return None
        return self.ordered_labels[idx + 1]

    def _disc_name_for_label(self, label):
        caudal = self._caudal_neighbor_label(label)
        if not caudal:
            return None
        return f"Disc_{label}_{caudal}"

    def _create_disc_fiducials(self):
        """Crea un fiducial celeste entre cada par de centros de cuerpo vertebral."""
        self._cleanup_disc_fiducials()
        node = self.scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SpineV3_Discos_Intervertebrales_Celestes_4mm")
        self.disc_fiducial_node = node
        self._disc_point_indices = {}
        self._add_node_to_scene_folder(node, "fiducials")

        try:
            node.CreateDefaultDisplayNodes()
        except Exception:
            pass
        dn = node.GetDisplayNode()
        if dn:
            for setter in ("SetColor", "SetSelectedColor"):
                try:
                    getattr(dn, setter)(*COLOR_DISC)
                except Exception:
                    pass
            # Fiduciales de discos: celestes, visibles y con tamaño real de 4 mm.
            # Según la versión de Slicer, el display node puede exponer SetGlyphSize
            # o solo SetGlyphScale; probamos ambos para compatibilidad.
            try:
                dn.SetGlyphTypeFromString("Sphere3D")
            except Exception:
                pass
            try:
                dn.SetGlyphSize(4.0)      # tamaño absoluto en mm, cuando está disponible
            except Exception:
                pass
            try:
                dn.SetGlyphScale(4.0)     # fallback compatible con versiones que usan escala
            except Exception:
                pass
            try:
                dn.SetTextScale(2.2)
            except Exception:
                pass
            try:
                dn.SetPointLabelsVisibility(True)
            except Exception:
                pass
            try:
                dn.SetVisibility(bool(self.show_disc_fiducials))
            except Exception:
                pass
            try:
                dn.SetVisibility2D(bool(self.show_disc_fiducials))
            except Exception:
                pass
            try:
                dn.SetVisibility3D(bool(self.show_disc_fiducials))
            except Exception:
                pass

        for label in self.ordered_labels:
            caudal = self._caudal_neighbor_label(label)
            if not caudal:
                continue
            p1 = self._get_current_pivot_world(label)
            p2 = self._get_current_pivot_world(caudal)
            disc = (np.array(p1, dtype=float) + np.array(p2, dtype=float)) * 0.5
            idx = self._add_fiducial_point(node, disc, self._disc_name_for_label(label))
            self._disc_point_indices[label] = idx

        print(f"[Discos] {len(self._disc_point_indices)} fiduciales celestes Disc_XX_YY creados entre centros de cuerpos vertebrales (glyph 4 mm).")

    def _set_disc_fiducials_visible(self, visible):
        self.show_disc_fiducials = bool(visible)
        if self.disc_fiducial_node and self.disc_fiducial_node.GetDisplayNode():
            dn = self.disc_fiducial_node.GetDisplayNode()
            dn.SetVisibility(bool(visible))
            try:
                dn.SetVisibility2D(bool(visible))
            except Exception:
                pass
            try:
                dn.SetVisibility3D(bool(visible))
            except Exception:
                pass

    def _read_disc_position_world(self, idx):
        node = self.disc_fiducial_node
        if not node:
            return None
        pos = [0.0, 0.0, 0.0]
        try:
            node.GetNthControlPointPositionWorld(idx, pos)
            return np.array(pos, dtype=float)
        except Exception:
            try:
                node.GetNthControlPointPosition(idx, pos)
                return np.array(pos, dtype=float)
            except Exception:
                return None

    def _recalculate_disc_fiducials_from_body_centers(self):
        """Recalcula Disc_XX_YY usando la posición ACTUAL de Pivot_XX y Pivot_YY."""
        if not self.disc_fiducial_node:
            self._create_disc_fiducials()
        node = self.disc_fiducial_node
        if not node:
            return 0

        # Si el usuario movió los centros del cuerpo, primero los tomamos como base.
        if self.pivot_fiducial_node:
            self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)

        updated = 0
        for label, idx in self._disc_point_indices.items():
            caudal = self._caudal_neighbor_label(label)
            if not caudal:
                continue
            p1 = self._get_current_pivot_world(label)
            p2 = self._get_current_pivot_world(caudal)
            disc = (np.array(p1, dtype=float) + np.array(p2, dtype=float)) * 0.5
            try:
                node.SetNthControlPointPositionWorld(idx, float(disc[0]), float(disc[1]), float(disc[2]))
            except Exception:
                try:
                    node.SetNthControlPointPosition(idx, float(disc[0]), float(disc[1]), float(disc[2]))
                except Exception:
                    continue
            updated += 1
        self._read_disc_fiducials_as_motion_pivots(update_status=False)
        return updated

    def _read_disc_fiducials_as_motion_pivots(self, update_status=False):
        """Lee Disc_XX_YY actuales y los convierte a pivots locales de la vértebra craneal XX."""
        if not self.disc_fiducial_node:
            return 0
        updated = 0
        for label, idx in self._disc_point_indices.items():
            pos = self._read_disc_position_world(idx)
            if pos is None:
                continue
            world = [float(pos[0]), float(pos[1]), float(pos[2]), 1.0]
            if label in self.transform_nodes:
                m = vtk.vtkMatrix4x4()
                inv = vtk.vtkMatrix4x4()
                self.transform_nodes[label].GetMatrixTransformToWorld(m)
                vtk.vtkMatrix4x4.Invert(m, inv)
                out = [0.0, 0.0, 0.0, 0.0]
                inv.MultiplyPoint(world, out)
                self._motion_pivots[label] = np.array(out[:3], dtype=float)
            else:
                self._motion_pivots[label] = np.array(world[:3], dtype=float)
            updated += 1
        if updated:
            self.use_disc_pivots = True
            self.live_disc_pivots = True
            if update_status:
                self._update_status(f"Pivotes discales activos: {updated} discos leídos.")
        return updated

    def _get_motion_pivot_local(self, label):
        """Pivot local usado para el movimiento: disco inferior si está activo, si no centro de cuerpo."""
        if self.use_disc_pivots and label in self._motion_pivots:
            return np.array(self._motion_pivots[label], dtype=float)
        return np.array(self._base_positions[label], dtype=float)


    # ── Construcción de transform: rotación + traslación ─────────────────────

    def _np_translation(self, v):
        m = np.eye(4, dtype=float)
        m[0:3, 3] = np.array(v[0:3], dtype=float)
        return m

    def _np_rotation_xyz(self, rx, ry, rz):
        ax = np.radians(rx)
        ay = np.radians(ry)
        az = np.radians(rz)
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)

        Rx = np.array([
            [1,  0,   0, 0],
            [0, cx, -sx, 0],
            [0, sx,  cx, 0],
            [0,  0,   0, 1],
        ], dtype=float)
        Ry = np.array([
            [ cy, 0, sy, 0],
            [  0, 1,  0, 0],
            [-sy, 0, cy, 0],
            [  0, 0,  0, 1],
        ], dtype=float)
        Rz = np.array([
            [cz, -sz, 0, 0],
            [sz,  cz, 0, 0],
            [ 0,   0, 1, 0],
            [ 0,   0, 0, 1],
        ], dtype=float)
        return Rz @ Ry @ Rx

    def _anchor_is_locked(self, label):
        """True si la vértebra inferior/ancla debe permanecer fija."""
        return bool(label == self.anchor_label and not self.anchor_motion_enabled)

    def _joint_matrix_np(self, label, rotation_scale=1.0):
        """Transformación elemental de una vértebra alrededor de su pivot actual.

        En modo cadena, esta matriz representa el movimiento del hueso/padre.
        Si C6 flexiona, la matriz de C6 se aplica también a C5, C4, C3... como
        una cadena cinemática, no como rotaciones locales independientes.
        """
        pivot = self._get_motion_pivot_local(label)
        rx, ry, rz = self._rotations[label]
        tx, ty, tz = self._translations[label]
        R = self._np_rotation_xyz(rx * rotation_scale, ry * rotation_scale, rz * rotation_scale)
        T = self._np_translation
        return T([tx, ty, tz]) @ T(pivot) @ R @ T(-pivot)

    def _build_transform_matrix(self, label):
        """
        Construye la matriz final.

        Modo nuevo, tipo Pose Mode/Bones:
        - Cada vértebra tiene un pivot editable, idealmente en el cuerpo vertebral.
        - La rotación de una vértebra caudal se propaga a las craneales.
        - Ejemplo: C6 flexiona hacia adelante -> C5, C4, C3 acompañan el arco
          generado por C6. No giran cada una solamente sobre su propio eje.

        Si se desactiva la cadena cinemática, vuelve al modo local anterior.
        """
        idx = self.ordered_labels.index(label)

        if self.kinematic_chain_enabled:
            # Composición tipo cadena de huesos usando la vértebra ancla elegida
            # por el usuario como raíz cinemática. ordered_labels va de craneal a
            # caudal: C1, C2, ... S5, COXIS.
            #
            # Si el label está craneal al ancla, hereda los joints desde el ancla
            # hacia arriba. Si está caudal al ancla, hereda los joints desde el
            # ancla hacia abajo. Esto permite cargar columna completa y elegir
            # C7, T12, L5, S1, COXIS, etc. como referencia.
            M = np.eye(4, dtype=float)
            anchor_idx = self.ordered_labels.index(self.anchor_label) if self.anchor_label in self.ordered_labels else len(self.ordered_labels)-1
            if idx <= anchor_idx:
                iterator = range(anchor_idx, idx - 1, -1)
            else:
                iterator = range(anchor_idx, idx + 1, 1)
            for k in iterator:
                lk = self.ordered_labels[k]
                if self._anchor_is_locked(lk):
                    continue
                scale = 1.0
                if k != idx:
                    scale = float(self.local_bend_fraction)
                M = M @ self._joint_matrix_np(lk, rotation_scale=scale)
        else:
            # Modo local clásico: cada modelo rota alrededor de su pivot propio.
            M = self._joint_matrix_np(label)

            # Conserva el pequeño desplazamiento FABRIK original si existe.
            if self.solver is not None:
                pivot = self._base_positions[label]
                ik_pos = self.solver.pos[idx]
                delta_ik = ik_pos - pivot
                M = self._np_translation(delta_ik) @ M

        vtk_m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_m.SetElement(r, c, float(M[r, c]))
        return vtk_m

    def _apply_all_transforms(self):
        self._dirty_transforms = False
        for label in self.ordered_labels:
            m = self._build_transform_matrix(label)
            self.transform_nodes[label].SetMatrixTransformToParent(m)
        self._update_pivot_fiducials()
        self._update_native_transform_interaction()

    def _schedule_transform_update(self):
        self._dirty_transforms = True
        try:
            if not self._flush_timer.isActive():
                self._flush_timer.start()
        except Exception:
            self._flush_transforms()

    def _flush_transforms(self):
        if not self._dirty_transforms:
            return
        self._apply_all_transforms()

    def _ensure_transforms_flushed(self):
        if self._dirty_transforms:
            try:
                self._flush_timer.stop()
            except Exception:
                pass
            self._flush_transforms()

    # ── Colisiones VTP ───────────────────────────────────────────────────────

    def _pair_key(self, a, b):
        ia = self.ordered_labels.index(a)
        ib = self.ordered_labels.index(b)
        return (a, b) if ia <= ib else (b, a)

    def _label_matrix_to_world(self, label):
        m = vtk.vtkMatrix4x4()
        self.transform_nodes[label].GetMatrixTransformToWorld(m)
        return m

    def _collision_source_node_for_label(self, label):
        # Si la vertebra fue drileada, usar la malla visual exacta para colision.
        # La proxy decimada puede suavizar/cerrar huecos y dar falsos contactos.
        if label in getattr(self, "_osteotomy_exact_collision_labels", set()):
            return self.model_nodes.get(label)
        node = self.collision_model_nodes.get(label) if self.collision_proxy_enabled else None
        return node or self.model_nodes.get(label)

    def _transformed_bounds(self, label, inflate=0.0):
        node = self._collision_source_node_for_label(label)
        poly = node.GetPolyData()
        if poly is None:
            return None
        b = [0.0] * 6
        poly.GetBounds(b)
        corners = [
            (b[0], b[2], b[4]), (b[0], b[2], b[5]),
            (b[0], b[3], b[4]), (b[0], b[3], b[5]),
            (b[1], b[2], b[4]), (b[1], b[2], b[5]),
            (b[1], b[3], b[4]), (b[1], b[3], b[5]),
        ]
        m = self._label_matrix_to_world(label)
        pts = []
        for p in corners:
            out = [0.0, 0.0, 0.0, 0.0]
            m.MultiplyPoint([float(p[0]), float(p[1]), float(p[2]), 1.0], out)
            pts.append(out[:3])
        arr = np.array(pts, dtype=float)
        mins = arr.min(axis=0) - float(inflate)
        maxs = arr.max(axis=0) + float(inflate)
        return (mins, maxs)

    def _bounds_overlap(self, a, b):
        if a is None or b is None:
            return False
        amin, amax = a
        bmin, bmax = b
        return bool(np.all(amax >= bmin) and np.all(bmax >= amin))

    def _transformed_polydata_for_collision(self, label, cache):
        if label in cache:
            return cache[label]
        node = self._collision_source_node_for_label(label)
        poly = node.GetPolyData()
        if poly is None:
            return None
        transform = vtk.vtkTransform()
        transform.SetMatrix(self._label_matrix_to_world(label))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(poly)
        tf.SetTransform(transform)
        tf.Update()
        out = tf.GetOutput()
        cache[label] = out
        return out

    def _sample_polydata_points(self, poly):
        n = poly.GetNumberOfPoints() if poly else 0
        if n <= 0:
            return []
        max_points = max(50, int(self.collision_max_sample_points))
        step = max(1, int(np.ceil(float(n) / float(max_points))))
        return [poly.GetPoint(i) for i in range(0, n, step)]

    def _directed_surface_distance(self, sample_poly, target_poly):
        if not sample_poly or not target_poly:
            return None
        pts = sample_poly.GetPoints()
        n = pts.GetNumberOfPoints() if pts else 0
        if n <= 0:
            return None
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(target_poly)
        max_points = max(50, int(self.collision_max_sample_points))
        step = max(1, int(np.ceil(float(n) / float(max_points))))
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            arr = vtk_to_numpy(pts.GetData())
            sample = arr[::step]
            if sample.size == 0:
                return None
            distances = np.empty((sample.shape[0],), dtype=float)
            for i, p in enumerate(sample):
                distances[i] = float(implicit.EvaluateFunction((float(p[0]), float(p[1]), float(p[2]))))
            return float(np.abs(distances).min()), float(distances.min())
        except Exception:
            min_abs = None
            min_signed = None
            for i in range(0, n, step):
                d = float(implicit.EvaluateFunction(pts.GetPoint(i)))
                ad = abs(d)
                min_abs = ad if min_abs is None else min(min_abs, ad)
                min_signed = d if min_signed is None else min(min_signed, d)
            if min_abs is None:
                return None
            return min_abs, min_signed

    def _collision_pair_info(self, a, b, cache=None):
        if a == b or a not in self.model_nodes or b not in self.model_nodes:
            return None
        cache = cache if cache is not None else {}
        margin = max(0.0, float(self.collision_margin_mm))
        ba = self._transformed_bounds(a, inflate=margin)
        bb = self._transformed_bounds(b, inflate=margin)
        if not self._bounds_overlap(ba, bb):
            return None
        pa = self._transformed_polydata_for_collision(a, cache)
        pb = self._transformed_polydata_for_collision(b, cache)
        da = self._directed_surface_distance(pa, pb)
        db = self._directed_surface_distance(pb, pa)
        values = [x for x in (da, db) if x is not None]
        if not values:
            return None
        min_abs = min(v[0] for v in values)
        min_signed = min(v[1] for v in values)
        penetration_eps = max(0.05, margin * 0.25)
        near_eps = min(max(0.12, margin * 0.35), 0.35)
        # V48: contacto real o casi-contacto muy estrecho. Evita falsos positivos amplios.
        if min_signed < -penetration_eps or min_abs <= near_eps:
            return {
                "a": a,
                "b": b,
                "min_abs_mm": float(min_abs),
                "min_signed_mm": float(min_signed),
                "penetration_mm": float(max(0.0, -min_signed)),
                "near_eps_mm": float(near_eps),
                "pair": self._pair_key(a, b),
            }
        return None

    def _collision_worse_than_baseline(self, info):
        pair = info.get("pair") or self._pair_key(info["a"], info["b"])
        base = self._collision_baseline_pairs.get(pair)
        if not base:
            return True
        margin = max(0.0, float(self.collision_margin_mm))
        # Si la distancia baja claramente o la penetración se vuelve más negativa,
        # se considera un empeoramiento respecto del contacto inicial permitido.
        distance_drop = float(base.get("min_abs_mm", 999.0)) - float(info["min_abs_mm"])
        penetration_drop = float(base.get("min_signed_mm", 999.0)) - float(info["min_signed_mm"])
        return bool(distance_drop > max(0.1, margin * 0.25) or penetration_drop > max(0.1, margin * 0.25))

    def _candidate_collision_pairs(self, moved_labels):
        labels = [l for l in moved_labels if l in self.ordered_labels]
        pairs = set()
        radius = max(1, int(self.collision_neighbor_radius))
        for label in labels:
            idx = self.ordered_labels.index(label)
            lo = max(0, idx - radius)
            hi = min(len(self.ordered_labels) - 1, idx + radius)
            for j in range(lo, hi + 1):
                other = self.ordered_labels[j]
                if other != label:
                    pairs.add(self._pair_key(label, other))
        return sorted(pairs, key=lambda p: (self.ordered_labels.index(p[0]), self.ordered_labels.index(p[1])))

    def _collision_probe_labels(self, moved_labels):
        """Amplía la zona revisada a vecinos craneales y caudales.

        La cinemática puede arrastrar principalmente hacia arriba, pero el contacto
        anatómico relevante puede ocurrir también contra la vértebra inferior.
        """
        labels = [l for l in moved_labels if l in self.ordered_labels]
        radius = max(1, int(self.collision_neighbor_radius))
        probe = set(labels)
        for label in labels:
            idx = self.ordered_labels.index(label)
            for j in range(max(0, idx - radius), min(len(self.ordered_labels) - 1, idx + radius) + 1):
                probe.add(self.ordered_labels[j])
        return sorted(probe, key=lambda l: self.ordered_labels.index(l))

    def _calibrate_collision_baseline(self):
        """Registra contactos ya presentes al inicio para no bloquear ni repintar la postura base."""
        self._sync_osteotomy_collision_proxies()
        self._collision_baseline_pairs = {}
        if not self.ordered_labels or not self.model_nodes:
            return
        cache = {}
        for pair in self._candidate_collision_pairs(self.ordered_labels):
            info = self._collision_pair_info(pair[0], pair[1], cache)
            if info:
                self._collision_baseline_pairs[pair] = {
                    "min_abs_mm": float(info["min_abs_mm"]),
                    "min_signed_mm": float(info["min_signed_mm"]),
                }
        if self._collision_baseline_pairs:
            self._log_event("WARN", f"Colisiones/contactos iniciales ignorados: {len(self._collision_baseline_pairs)} pares.")

    def _check_collisions_after_move(self, moved_labels):
        if not self.collision_enabled:
            return None
        self._sync_osteotomy_collision_proxies()
        cache = {}
        for a, b in self._candidate_collision_pairs(moved_labels):
            info = self._collision_pair_info(a, b, cache)
            if info and self._collision_worse_than_baseline(info):
                self._last_collision = info
                return info
        self._last_collision = None
        return None

    def _find_current_contact_after_move(self, moved_labels):
        hits = self._find_current_contacts_after_move(moved_labels)
        return hits[0] if hits else None

    def _find_current_contacts_after_move(self, moved_labels):
        if not self.collision_enabled:
            return []
        self._sync_osteotomy_collision_proxies()
        cache = {}
        hits = []
        for a, b in self._candidate_collision_pairs(moved_labels):
            info = self._collision_pair_info(a, b, cache)
            if info and self._collision_worse_than_baseline(info):
                hits.append(info)
        self._last_collision = hits[0] if hits else None
        return hits

    def _ensure_collision_color_node(self):
        if self._collision_color_node and self.scene.GetNodeByID(self._collision_color_node.GetID()):
            return self._collision_color_node
        node = self.scene.AddNewNodeByClass("vtkMRMLColorTableNode", "SpineV3_CollisionHeat")
        self._add_node_to_scene_folder(node, "colors")
        try:
            node.SetTypeToUser()
            node.SetNumberOfColors(256)
            for i in range(256):
                t = float(i) / 255.0
                # Negro/azul -> rojo -> amarillo/blanco.
                if t < 0.5:
                    r = t * 2.0
                    g = 0.0
                    b = 1.0 - t * 2.0
                else:
                    r = 1.0
                    g = (t - 0.5) * 2.0
                    b = 0.0
                if t > 0.85:
                    g = 1.0
                    b = (t - 0.85) / 0.15
                node.SetColor(i, f"heat_{i}", r, g, b, 1.0)
        except Exception:
            pass
        self._collision_color_node = node
        return node

    def _set_model_scalar_display(self, label, enabled):
        node = self.model_nodes.get(label)
        if not node:
            return
        dn = node.GetDisplayNode()
        if not dn:
            return
        try:
            dn.SetScalarVisibility(bool(enabled))
        except Exception:
            pass
        if enabled:
            color_node = self._ensure_collision_color_node()
            try:
                dn.SetAndObserveColorNodeID(color_node.GetID())
            except Exception:
                pass
            self._safe_call(dn, ["SetActiveScalarName", "SetActiveAttributeName"], "SpineCollisionHeat")
            self._safe_call(dn, ["SetScalarRangeFlag"], 0)
            self._safe_call(dn, ["SetScalarRange"], 0.0, 1.0)

    def _clear_collision_heatmap(self):
        for node in list(self._collision_overlay_nodes):
            try:
                if node and self.scene.GetNodeByID(node.GetID()):
                    self.scene.RemoveNode(node)
            except Exception:
                pass
        self._collision_overlay_nodes = []
        self._contact_mark_nodes_by_pair = {}
        self._contact_heat_centers_by_label = {}
        for label in list(self._collision_heat_labels):
            self._set_model_scalar_display(label, False)
        self._collision_heat_labels = set()
        self._paint_all_normal()

    def _remove_contact_mark_for_pair(self, pair):
        nodes = self._contact_mark_nodes_by_pair.get(pair, [])
        for node in list(nodes):
            try:
                if node and self.scene.GetNodeByID(node.GetID()):
                    self.scene.RemoveNode(node)
            except Exception:
                pass
            try:
                if node in self._collision_overlay_nodes:
                    self._collision_overlay_nodes.remove(node)
            except Exception:
                pass
        self._contact_mark_nodes_by_pair.pop(pair, None)
        for label in pair:
            self._contact_heat_centers_by_label.pop(label, None)
            self._set_model_scalar_display(label, False)

    def _clear_contact_marks_for_label(self, label):
        if not label:
            return
        pairs = [pair for pair in list(self._contact_mark_nodes_by_pair.keys()) if label in pair]
        for pair in pairs:
            self._remove_contact_mark_for_pair(pair)
        self._contact_heat_centers_by_label.pop(label, None)
        self._set_model_scalar_display(label, False)
        if label in self._collision_heat_labels:
            try:
                self._collision_heat_labels.remove(label)
            except Exception:
                pass

    def _apply_collision_heat_to_label(self, label, other_label, cache):
        mode = str(getattr(self, "collision_heatmap_mode", "PATCH")).upper()
        if mode == "SURFACE":
            return self._apply_collision_surface_heat_to_label(label, other_label, cache)
        if mode == "SPHERES":
            return self._apply_collision_sphere_marks_to_label(label, other_label, cache)
        return self._apply_collision_surface_patch_to_label(label, other_label, cache)

    def _collision_hot_points_for_label(self, label, other_label, cache, sampled=True):
        node = self.model_nodes.get(label)
        other_poly = self._transformed_polydata_for_collision(other_label, cache)
        if not node or not node.GetPolyData() or other_poly is None:
            return None, None, []
        poly = node.GetPolyData()
        n = poly.GetNumberOfPoints()
        if n <= 0:
            return node, poly, []

        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(other_poly)
        m = self._label_matrix_to_world(label)
        radius = max(0.1, float(self.collision_heatmap_radius_mm))
        if sampled:
            max_heat_points = max(100, int(self.collision_heatmap_max_points))
            step = max(1, int(np.ceil(float(n) / float(max_heat_points))))
        else:
            step = 1

        hot = []
        for i in range(0, n, step):
            p = poly.GetPoint(i)
            out = [0.0, 0.0, 0.0, 0.0]
            m.MultiplyPoint([float(p[0]), float(p[1]), float(p[2]), 1.0], out)
            d = float(implicit.EvaluateFunction(out[:3]))
            # V48: pintar penetracion y una banda estrecha de casi-contacto.
            near_band = min(max(0.12, float(self.collision_margin_mm) * 0.35), 0.35)
            if d < 0.0:
                heat = min(1.0, 0.55 + ((-float(d)) / radius) * 0.45)
            elif d <= near_band:
                heat = max(0.0, 0.55 * (1.0 - (float(d) / near_band)))
            else:
                continue
            if heat > 0.05:
                hot.append((float(heat), int(i), tuple(p)))
        return node, poly, hot

    def _remember_contact_heat_center(self, label, hot):
        if not hot:
            self._contact_heat_centers_by_label.pop(label, None)
            return
        total = sum(float(h[0]) for h in hot)
        if total <= 0.0:
            return
        acc = np.zeros(3, dtype=float)
        for heat, _idx, pos in hot:
            acc += np.array(pos, dtype=float) * float(heat)
        self._contact_heat_centers_by_label[label] = acc / float(total)

    def _apply_collision_surface_heat_to_label(self, label, other_label, cache):
        node, poly, hot = self._collision_hot_points_for_label(label, other_label, cache, sampled=True)
        other_poly = self._transformed_polydata_for_collision(other_label, cache)
        if not node or not poly or other_poly is None:
            return None
        n = poly.GetNumberOfPoints()
        if not hot:
            try:
                poly.GetPointData().RemoveArray("SpineCollisionHeat")
                poly.Modified()
                node.Modified()
            except Exception:
                pass
            self._contact_heat_centers_by_label.pop(label, None)
            self._set_model_scalar_display(label, False)
            return None

        heat_values = np.zeros((n,), dtype=float)
        hot.sort(key=lambda x: x[0], reverse=True)
        seeds = hot[:min(90, len(hot))]
        radius = max(0.1, float(self.collision_heatmap_radius_mm))
        spread_radius = max(0.8, radius * 1.25)

        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(other_poly)
        m = self._label_matrix_to_world(label)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()
        ids = vtk.vtkIdList()

        for seed_heat, _idx, pos in seeds:
            ids.Reset()
            locator.FindPointsWithinRadius(float(spread_radius), pos, ids)
            pos_arr = np.array(pos, dtype=float)
            for j in range(ids.GetNumberOfIds()):
                pid = ids.GetId(j)
                p = poly.GetPoint(pid)
                p_arr = np.array(p, dtype=float)
                dist_seed = float(np.linalg.norm(p_arr - pos_arr))
                falloff = max(0.0, 1.0 - (dist_seed / spread_radius))
                if falloff <= 0.0:
                    continue

                out = [0.0, 0.0, 0.0, 0.0]
                m.MultiplyPoint([float(p[0]), float(p[1]), float(p[2]), 1.0], out)
                d = float(implicit.EvaluateFunction(out[:3]))
                near_band = min(max(0.12, float(self.collision_margin_mm) * 0.35), 0.35)
                if d < 0.0:
                    true_heat = min(1.0, 0.55 + ((-float(d)) / radius) * 0.45)
                elif d <= near_band:
                    true_heat = max(0.0, 0.55 * (1.0 - (float(d) / near_band)))
                else:
                    continue
                value = min(float(seed_heat), float(true_heat)) * (falloff ** 0.35)
                if value > heat_values[pid]:
                    heat_values[pid] = value

        heat_values[heat_values < 0.05] = 0.0

        heat_array = vtk.vtkFloatArray()
        heat_array.SetName("SpineCollisionHeat")
        heat_array.SetNumberOfComponents(1)
        heat_array.SetNumberOfTuples(n)
        for i, value in enumerate(heat_values):
            heat_array.SetValue(i, float(value))

        pd = poly.GetPointData()
        old = pd.GetArray("SpineCollisionHeat")
        if old is not None:
            pd.RemoveArray("SpineCollisionHeat")
        pd.AddArray(heat_array)
        pd.SetActiveScalars("SpineCollisionHeat")
        poly.Modified()
        node.Modified()
        self._set_model_scalar_display(label, True)
        self._collision_heat_labels.add(label)
        self._remember_contact_heat_center(label, seeds)
        return None

    def _apply_collision_surface_patch_to_label(self, label, other_label, cache):
        node, poly, hot = self._collision_hot_points_for_label(label, other_label, cache, sampled=True)
        if not node or not poly or not hot:
            return None
        self._set_model_scalar_display(label, False)
        self._remember_contact_heat_center(label, hot)

        hot.sort(key=lambda x: x[0], reverse=True)
        hot = hot[:140]

        try:
            poly.BuildLinks()
        except Exception:
            pass
        selected_cells = set()
        ids = vtk.vtkIdList()
        for _heat, pid, _pos in hot:
            ids.Reset()
            try:
                poly.GetPointCells(int(pid), ids)
            except Exception:
                continue
            for j in range(ids.GetNumberOfIds()):
                selected_cells.add(int(ids.GetId(j)))
            if len(selected_cells) >= 260:
                break

        if not selected_cells:
            return None

        out_pts = vtk.vtkPoints()
        out_polys = vtk.vtkCellArray()
        lift = 0.08
        for cid in selected_cells:
            cell = poly.GetCell(int(cid))
            if cell is None:
                continue
            pids = cell.GetPointIds()
            nids = pids.GetNumberOfIds()
            if nids < 3:
                continue
            base = [np.array(poly.GetPoint(pids.GetId(k)), dtype=float) for k in range(nids)]
            normal = np.cross(base[1] - base[0], base[2] - base[0])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                normal = normal / norm
            else:
                normal = np.zeros(3, dtype=float)
            new_ids = vtk.vtkIdList()
            for p in base:
                q = p + normal * lift
                new_ids.InsertNextId(out_pts.InsertNextPoint(float(q[0]), float(q[1]), float(q[2])))
            out_polys.InsertNextCell(new_ids)

        patch_poly = vtk.vtkPolyData()
        patch_poly.SetPoints(out_pts)
        patch_poly.SetPolys(out_polys)
        if patch_poly.GetNumberOfCells() <= 0:
            return None

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(patch_poly)
        clean.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        overlay_poly = vtk.vtkPolyData()
        overlay_poly.DeepCopy(normals.GetOutput())
        overlay = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
        overlay.SetName(f"SpineContactPatch_{label}_{other_label}")
        overlay.SetAndObservePolyData(overlay_poly)
        if self.contact_marks_follow_vertebra and label in self.transform_nodes:
            overlay.SetAndObserveTransformNodeID(self.transform_nodes[label].GetID())
        overlay.CreateDefaultDisplayNodes()
        self._add_node_to_scene_folder(overlay, "contacts")
        dn = overlay.GetDisplayNode()
        if dn:
            dn.SetColor(1.0, 0.03, 0.0)
            dn.SetOpacity(0.92)
            dn.SetVisibility(True)
            self._safe_call(dn, "SetVisibility3D", True)
            self._safe_call(dn, "SetPickable", False)
            self._safe_call(dn, "SetBackfaceCulling", False)
        try:
            overlay.SetSelectable(0)
        except Exception:
            pass
        self._collision_overlay_nodes.append(overlay)
        self._collision_heat_labels.add(label)
        return overlay

    def _apply_collision_sphere_marks_to_label(self, label, other_label, cache):
        node, poly, hot = self._collision_hot_points_for_label(label, other_label, cache, sampled=True)
        if not node or not poly or not hot:
            return None
        self._set_model_scalar_display(label, False)
        self._remember_contact_heat_center(label, hot)

        hot.sort(key=lambda x: x[0], reverse=True)
        hot = hot[:120]

        append = vtk.vtkAppendPolyData()
        for heat, _idx, pos in hot:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(float(pos[0]), float(pos[1]), float(pos[2]))
            sphere.SetRadius(0.45 + 0.75 * float(heat))
            sphere.SetThetaResolution(8)
            sphere.SetPhiResolution(8)
            sphere.Update()
            append.AddInputData(sphere.GetOutput())
        append.Update()

        overlay_poly = vtk.vtkPolyData()
        overlay_poly.DeepCopy(append.GetOutput())
        overlay = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
        overlay.SetName(f"SpineContactOverlay_{label}_{other_label}")
        overlay.SetAndObservePolyData(overlay_poly)
        if self.contact_marks_follow_vertebra and label in self.transform_nodes:
            overlay.SetAndObserveTransformNodeID(self.transform_nodes[label].GetID())
        overlay.CreateDefaultDisplayNodes()
        self._add_node_to_scene_folder(overlay, "contacts")
        dn = overlay.GetDisplayNode()
        if dn:
            dn.SetColor(1.0, 0.12, 0.0)
            dn.SetOpacity(0.95)
            dn.SetVisibility(True)
            self._safe_call(dn, "SetVisibility3D", True)
            self._safe_call(dn, "SetPickable", False)
        try:
            overlay.SetSelectable(0)
        except Exception:
            pass
        self._collision_overlay_nodes.append(overlay)
        self._collision_heat_labels.add(label)
        return overlay

    def _apply_collision_heatmap(self, hit):
        if not self.collision_heatmap_enabled or not hit:
            return
        cache = {}
        self._apply_collision_heat_to_label(hit["a"], hit["b"], cache)
        self._apply_collision_heat_to_label(hit["b"], hit["a"], cache)

    def _apply_collision_heatmaps(self, hits):
        if not self.collision_heatmap_enabled or not hits:
            return
        cache = {}
        for hit in hits:
            pair = tuple(hit.get("pair") or self._pair_key(hit["a"], hit["b"]))
            if self.contact_marks_persistent:
                self._remove_contact_mark_for_pair(pair)
            else:
                self._clear_collision_heatmap()
            # V3_57: la transparencia es opcional e independiente del heatmap.
            # Solo se aplica si el usuario la activó explícitamente en el panel.
            nodes = []
            node_a = self._apply_collision_heat_to_label(hit["a"], hit["b"], cache)
            node_b = self._apply_collision_heat_to_label(hit["b"], hit["a"], cache)
            if node_a:
                nodes.append(node_a)
            if node_b:
                nodes.append(node_b)
            if self.contact_marks_persistent:
                self._contact_mark_nodes_by_pair[pair] = nodes

    def _reject_move_if_collision(self, moved_labels):
        if self.collision_blocking_enabled:
            hit = self._check_collisions_after_move(moved_labels)
            hits = [hit] if hit else []
        else:
            hits = self._find_current_contacts_after_move(moved_labels)
            hit = hits[0] if hits else None
        if not hits:
            return False
        self._apply_collision_heatmaps(hits)
        if not self.collision_blocking_enabled:
            pairs = ", ".join([f"{h['a']}-{h['b']} ({h['min_abs_mm']:.2f} mm)" for h in hits[:3]])
            if len(hits) > 3:
                pairs += f" +{len(hits)-3}"
            msg = f"Contacto detectado: {pairs}. Movimiento libre."
            self._update_status(msg)
            return False
        self._revert_to_snapshot()
        self._apply_all_transforms()
        self._sync_rot_sliders()
        self._sync_trans_sliders()
        self._update_pivot_fiducials()
        self._update_native_transform_interaction()
        msg = (f"Colisión detectada: {hit['a']} con {hit['b']} "
               f"(distancia {hit['min_abs_mm']:.2f} mm). Movimiento cancelado.")
        self._update_status(msg)
        self._log_event("WARN", msg)
        return True

    # ── Snapshot para revertir ────────────────────────────────────────────────

    def _save_snapshot(self):
        self._ensure_transforms_flushed()
        self._prev_solver_pos = [p.copy() for p in self.solver.pos]
        self._prev_rotations = {l: list(v) for l, v in self._rotations.items()}
        self._prev_translations = {l: list(v) for l, v in self._translations.items()}
        self._prev_matrices_flat = {}
        self._prev_transforms = self._prev_matrices_flat
        for label in self.ordered_labels:
            m = vtk.vtkMatrix4x4()
            self.transform_nodes[label].GetMatrixTransformToParent(m)
            self._prev_matrices_flat[label] = [m.GetElement(r, c) for r in range(4) for c in range(4)]

    def _revert_to_snapshot(self):
        if self._prev_solver_pos:
            self.solver.pos = [p.copy() for p in self._prev_solver_pos]
        if self._prev_rotations:
            self._rotations = {l: list(v) for l, v in self._prev_rotations.items()}
        if self._prev_translations:
            self._translations = {l: list(v) for l, v in self._prev_translations.items()}
        for label, flat in self._prev_matrices_flat.items():
            m = vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    m.SetElement(r, c, float(flat[r * 4 + c]))
            self.transform_nodes[label].SetMatrixTransformToParent(m)
        self._dirty_transforms = False

    def _harmonic_weight(self, dist):
        """Peso suave tipo campana para que el movimiento no sea escalonado.

        Antes se usaba decay**distancia, que generaba saltos bruscos entre
        vértebra activa, primera vecina y segunda vecina. Esta función mezcla
        una ventana coseno con el peso de vecina para obtener una caída más
        armónica y continua.
        """
        if dist <= 0:
            return 1.0
        radius = max(1, int(self.influence_radius))
        decay = max(0.0, min(1.0, float(self.influence_decay)))
        t = min(1.0, float(dist) / float(radius + 1))
        cosine_window = 0.5 * (1.0 + np.cos(np.pi * t))
        return float(cosine_window * (decay ** dist))

    def _influence_items(self, center_label):
        """
        Devuelve [(label, distancia, peso)] para la vértebra activa y sus vecinas.
        La activa pesa 1.0; las vecinas caen con una curva armónica, no por
        escalones rígidos.
        """
        if center_label not in self.ordered_labels:
            return []
        center_idx = self.ordered_labels.index(center_label)
        if not self.dynamic_enabled:
            return [(center_label, 0, 1.0)]

        items = []
        radius = max(0, int(self.influence_radius))
        for j, label in enumerate(self.ordered_labels):
            dist = abs(j - center_idx)
            if dist > radius:
                continue
            if self._anchor_is_locked(label):
                # El ancla queda fija salvo que el usuario active su edición.
                continue
            weight = self._harmonic_weight(dist)
            if weight <= 0.0001:
                continue
            items.append((label, dist, weight))
        items.sort(key=lambda x: (x[1], self.ordered_labels.index(x[0])))
        return items

    def _rotation_offset_for_solver(self, label):
        """Convierte la rotación acumulada de una vértebra en un pequeño objetivo espacial para FABRIK.

        Mantiene la idea original del programa, pero ahora se aplica a todas las
        vértebras influenciadas, no solo a la activa.
        """
        idx = self.ordered_labels.index(label)
        seg_idx = max(min(idx - 1, len(self.solver.seg_lens) - 1), 0)
        seg_len = self.solver.seg_lens[seg_idx] if self.solver.seg_lens else 1.0
        rx, ry, rz = self._rotations[label]
        rad_z = np.radians(rz)
        rad_x = np.radians(rx)
        offset = np.array([
            np.sin(rad_z) * seg_len * 0.5,
            0.0,
            np.sin(rad_x) * seg_len * 0.5,
        ], dtype=float)
        return offset

    def _solve_distributed_targets(self, influenced_labels):
        """Actualiza FABRIK para varias vértebras influenciadas.

        Se llama después de modificar rotaciones/traslaciones. La activa y las
        vecinas reciben targets parciales, haciendo que el movimiento no quede
        rígido en una sola vértebra.
        """
        if not influenced_labels:
            return
        # Resolver de caudal a craneal ayuda a estabilizar cuando el ancla está abajo.
        labels = sorted(set(influenced_labels), key=lambda l: self.ordered_labels.index(l), reverse=True)
        for l in labels:
            if self._anchor_is_locked(l):
                continue
            idx = self.ordered_labels.index(l)
            base = self._base_positions[l]
            tx, ty, tz = self._translations[l]
            target = base + self._rotation_offset_for_solver(l) + np.array([tx, ty, tz], dtype=float)
            self.solver.solve(idx, target)

    # ── Movimiento principal ──────────────────────────────────────────────────

    def apply_rotation(self, label, drx=0.0, dry=0.0, drz=0.0):
        """
        Aplica un delta de rotación a 'label' y lo distribuye a las vértebras
        vecinas con un peso decreciente. Si la vértebra activa está entre dos,
        ambas vecinas acompañan el movimiento.
        """
        if self._anchor_is_locked(label):
            self._update_status(f"{label} es el ancla fija. Activá 'Permitir mover ancla' para editarla.")
            return
        if self.pivot_mode == "MANUAL_FIDUCIALS" or self.manual_pivots_enabled or self.live_fiducial_pivots:
            self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)
        if self.use_disc_pivots and self.live_disc_pivots:
            self._read_disc_fiducials_as_motion_pivots(update_status=False)
        self._save_snapshot()

        active_idx = self.ordered_labels.index(label)

        if self.kinematic_chain_enabled:
            # La rotación principal queda en la vértebra activa. Esa rotación,
            # por composición jerárquica, arrastra a todas las vértebras craneales.
            # No se reparte como rotación local fuerte en C5/C4/C3, porque eso era
            # lo que hacía que giraran sobre su propio eje.
            self._rotations[label][0] += drx
            self._rotations[label][1] += dry
            self._rotations[label][2] += drz

            influenced = []
            radius = max(0, int(self.influence_radius))
            for j in range(active_idx, -1, -1):
                l = self.ordered_labels[j]
                dist = active_idx - j
                if dist > radius:
                    continue
                # weight se informa para status; el arrastre padre-hijo real lo
                # hace _build_transform_matrix con la matriz de la vértebra activa.
                weight = self._harmonic_weight(dist)
                influenced.append((l, dist, weight))

            # Curvatura local leve opcional: agrega un pequeño delta propio a las
            # vértebras craneales, pero el movimiento grande sigue siendo el arco
            # proyectado por la vértebra inferior.
            if self.local_bend_fraction > 0.0 and self.dynamic_enabled:
                for l, dist, weight in influenced:
                    if l == label or self._anchor_is_locked(l):
                        continue
                    local_w = float(self.local_bend_fraction) * float(weight)
                    self._rotations[l][0] += drx * local_w
                    self._rotations[l][1] += dry * local_w
                    self._rotations[l][2] += drz * local_w
        else:
            influenced = self._influence_items(label)
            changed_labels = []
            for l, dist, weight in influenced:
                if self._anchor_is_locked(l):
                    continue
                self._rotations[l][0] += drx * weight
                self._rotations[l][1] += dry * weight
                self._rotations[l][2] += drz * weight
                changed_labels.append(l)
            self._solve_distributed_targets(changed_labels)

        self._schedule_transform_update()
        if self.collision_enabled:
            self._ensure_transforms_flushed()
        moved_labels = [label]
        if self._reject_move_if_collision(moved_labels):
            return
        moved = ", ".join([f"{l}({w:.2f})" for l, d, w in influenced])
        if not self._last_collision:
            self._update_status(f"Movimiento armónico desde {label}: {moved} | sin contacto")

    def apply_translation(self, label, dx=0.0, dy=0.0, dz=0.0):
        """Traslación manual distribuida a vecinas.

        El slider indica la traslación absoluta de la vértebra activa. Internamente
        se calcula el delta y ese delta se reparte hacia las vecinas.
        """
        if self._anchor_is_locked(label):
            self._update_status(f"{label} es el ancla fija. Activá 'Permitir mover ancla' para editarla.")
            return
        if self.pivot_mode == "MANUAL_FIDUCIALS" or self.manual_pivots_enabled or self.live_fiducial_pivots:
            self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)
        if self.use_disc_pivots and self.live_disc_pivots:
            self._read_disc_fiducials_as_motion_pivots(update_status=False)
        self._save_snapshot()

        old = np.array(self._translations[label], dtype=float)
        new_value = np.array([dx, dy, dz], dtype=float)
        delta = new_value - old

        active_idx = self.ordered_labels.index(label)
        if self.kinematic_chain_enabled:
            # La traslación del hueso activo también arrastra a la cadena craneal
            # por composición jerárquica. No se copia como traslación independiente
            # a cada vértebra, evitando movimientos desarmónicos.
            self._translations[label][0] += float(delta[0])
            self._translations[label][1] += float(delta[1])
            self._translations[label][2] += float(delta[2])
            influenced = []
            radius = max(0, int(self.influence_radius))
            for j in range(active_idx, -1, -1):
                l = self.ordered_labels[j]
                dist = active_idx - j
                if dist > radius:
                    continue
                influenced.append((l, dist, self._harmonic_weight(dist)))
        else:
            influenced = self._influence_items(label)
            changed_labels = []
            for l, dist, weight in influenced:
                if self._anchor_is_locked(l):
                    continue
                self._translations[l][0] += float(delta[0] * weight)
                self._translations[l][1] += float(delta[1] * weight)
                self._translations[l][2] += float(delta[2] * weight)
                changed_labels.append(l)
            self._solve_distributed_targets(changed_labels)

        self._schedule_transform_update()
        if self.collision_enabled:
            self._ensure_transforms_flushed()
        moved_labels = [label]
        if self._reject_move_if_collision(moved_labels):
            return
        moved = ", ".join([f"{l}({w:.2f})" for l, d, w in influenced])
        if not self._last_collision:
            self._update_status(f"Traslación armónica desde {label}: {moved} | sin contacto")

    def reset_vertebra(self, label):
        self._sync_osteotomy_collision_proxies()
        self._clear_collision_heatmap()
        self._rotations[label]    = [0.0, 0.0, 0.0]
        self._translations[label] = [0.0, 0.0, 0.0]
        idx = self.ordered_labels.index(label)
        self.solver.pos[idx] = self._base_positions[label].copy()
        self._apply_all_transforms()
        self._reset_widgets()

    def reset_all(self):
        # V3_57: restaurar opacidad de cualquier modelo que haya quedado transparente
        for label in list(getattr(self, "_collision_transparent_labels", set())):
            self._collision_transparent_labels = set()

        self._sync_osteotomy_collision_proxies()
        self._clear_collision_heatmap()
        for l in self.ordered_labels:
            self._rotations[l]    = [0.0, 0.0, 0.0]
            self._translations[l] = [0.0, 0.0, 0.0]
        self.solver.reset()
        self._apply_all_transforms()
        self._update_pivot_fiducials()
        if self.use_disc_pivots:
            self._read_disc_fiducials_as_motion_pivots(update_status=False)
        self._calibrate_collision_baseline()
        self._reset_widgets()
        print(f"[SpineSimulator {self.version}] Columna restaurada.")

    # ── Osteotomía virtual VTP ───────────────────────────────────────────────

    def _remove_osteotomy_preview(self):
        if self._osteotomy_preview_node:
            try:
                if self.scene.GetNodeByID(self._osteotomy_preview_node.GetID()):
                    self.scene.RemoveNode(self._osteotomy_preview_node)
            except Exception:
                pass
        self._osteotomy_preview_node = None
        self._osteotomy_preview_label = None

    def _contact_center_local_for_label(self, label):
        """Busca marca o centro de calor persistente pegado a la vertebra."""
        if not label:
            return None
        center = self._contact_heat_centers_by_label.get(label)
        if center is not None:
            return np.array(center, dtype=float)
        for pair, nodes in list(self._contact_mark_nodes_by_pair.items()):
            if label not in pair:
                continue
            for node in nodes:
                if not node or not node.GetPolyData():
                    continue
                name = node.GetName() or ""
                if not (name.startswith(f"SpineContactOverlay_{label}_") or name.startswith(f"SpineContactPatch_{label}_")):
                    continue
                b = [0.0] * 6
                node.GetPolyData().GetBounds(b)
                return np.array([
                    (b[0] + b[1]) * 0.5,
                    (b[2] + b[3]) * 0.5,
                    (b[4] + b[5]) * 0.5,
                ], dtype=float)
        return None

    def _set_osteotomy_center_from_contact(self):
        label = self.active_label
        if not label or label not in self.model_nodes:
            self._update_status("Seleccioná una vértebra antes de ubicar la broca.")
            return
        center = self._contact_center_local_for_label(label)
        if center is None:
            center = np.array(self._get_motion_pivot_local(label), dtype=float)
            self._update_status(f"Sin marca de contacto en {label}; broca colocada en el pivot.")
        else:
            self._update_status(f"Broca colocada sobre marca de contacto de {label}.")
        self._osteotomy_center_local[label] = center
        self._update_osteotomy_preview(label)

    def _update_osteotomy_preview(self, label=None):
        label = label or self.active_label
        if not label or label not in self.model_nodes:
            return
        center = self._osteotomy_center_local.get(label)
        if center is None:
            center = np.array(self._get_motion_pivot_local(label), dtype=float)
            self._osteotomy_center_local[label] = center
        self._remove_osteotomy_preview()

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
        sphere.SetRadius(float(self.osteotomy_radius_mm))
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(16)
        sphere.Update()

        node = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
        node.SetName(f"SpineOsteotomyPreview_{label}")
        node.SetAndObservePolyData(sphere.GetOutput())
        if label in self.transform_nodes:
            node.SetAndObserveTransformNodeID(self.transform_nodes[label].GetID())
        node.CreateDefaultDisplayNodes()
        self._add_node_to_scene_folder(node, "osteotomy")
        dn = node.GetDisplayNode()
        if dn:
            dn.SetColor(1.0, 0.15, 0.05)
            dn.SetOpacity(0.35)
            dn.SetVisibility(True)
            self._safe_call(dn, "SetPickable", False)
        try:
            node.SetSelectable(0)
        except Exception:
            pass
        self._osteotomy_preview_node = node
        self._osteotomy_preview_label = label

    def _cell_center(self, poly, cell):
        ids = cell.GetPointIds()
        n = ids.GetNumberOfIds()
        if n <= 0:
            return None
        acc = np.zeros(3, dtype=float)
        for i in range(n):
            acc += np.array(poly.GetPoint(ids.GetId(i)), dtype=float)
        return acc / float(n)

    def _clip_polydata_by_local_sphere(self, poly, center, radius):
        """Recorte VTP simple: elimina celdas cuyo centro cae dentro de la broca."""
        if poly is None or poly.GetNumberOfCells() == 0:
            return None, 0
        center = np.array(center, dtype=float)
        r2 = float(radius) ** 2

        out = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.DeepCopy(poly.GetPoints())
        out.SetPoints(pts)

        new_polys = vtk.vtkCellArray()
        removed = 0
        for cid in range(poly.GetNumberOfCells()):
            cell = poly.GetCell(cid)
            c = self._cell_center(poly, cell)
            if c is not None and float(np.sum((c - center) ** 2)) <= r2:
                removed += 1
                continue
            ids = cell.GetPointIds()
            id_list = vtk.vtkIdList()
            for i in range(ids.GetNumberOfIds()):
                id_list.InsertNextId(ids.GetId(i))
            new_polys.InsertNextCell(id_list)
        out.SetPolys(new_polys)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(out)
        clean.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        return result, removed

    def _clip_polydata_by_local_sphere_fast(self, poly, center, radius):
        if poly is None or poly.GetNumberOfCells() == 0:
            return None, 0
        before_cells = int(poly.GetNumberOfCells())
        sphere = vtk.vtkSphere()
        sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
        sphere.SetRadius(float(radius))

        extract = vtk.vtkExtractPolyDataGeometry()
        extract.SetInputData(poly)
        extract.SetImplicitFunction(sphere)
        extract.ExtractInsideOff()
        extract.PassPointsOff()
        extract.Update()

        result = extract.GetOutput()
        if result is None or result.GetNumberOfCells() <= 0:
            return None, 0
        removed = max(0, before_cells - int(result.GetNumberOfCells()))
        if removed <= 0:
            return None, 0

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(extract.GetOutputPort())
        clean.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(normals.GetOutput())
        return out, removed

    def _clip_polydata_by_local_spheres_fast(self, poly, centers, radius):
        if poly is None or poly.GetNumberOfCells() == 0 or not centers:
            return None, 0
        before_cells = int(poly.GetNumberOfCells())
        boolean = vtk.vtkImplicitBoolean()
        boolean.SetOperationTypeToUnion()
        for center in centers:
            sphere = vtk.vtkSphere()
            sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
            sphere.SetRadius(float(radius))
            boolean.AddFunction(sphere)

        extract = vtk.vtkExtractPolyDataGeometry()
        extract.SetInputData(poly)
        extract.SetImplicitFunction(boolean)
        extract.ExtractInsideOff()
        extract.PassPointsOff()
        extract.Update()

        result = extract.GetOutput()
        if result is None or result.GetNumberOfCells() <= 0:
            return None, 0
        removed = max(0, before_cells - int(result.GetNumberOfCells()))
        if removed <= 0:
            return None, 0

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(extract.GetOutputPort())
        clean.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(normals.GetOutput())
        return out, removed

    def _world_to_label_local(self, label, world_pos):
        m = vtk.vtkMatrix4x4()
        inv = vtk.vtkMatrix4x4()
        self.transform_nodes[label].GetMatrixTransformToWorld(m)
        vtk.vtkMatrix4x4.Invert(m, inv)
        out = [0.0, 0.0, 0.0, 0.0]
        inv.MultiplyPoint([float(world_pos[0]), float(world_pos[1]), float(world_pos[2]), 1.0], out)
        return np.array(out[:3], dtype=float)

    def _pick_active_surface_world(self, x, y):
        """Devuelve el punto RAS/world sobre la malla visual activa bajo el mouse."""
        label = self.active_label
        if not label or label not in self.model_nodes:
            return None
        try:
            renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.01)
            ok = picker.Pick(int(x), int(y), 0, renderer)
            if not ok:
                return None
            actor = picker.GetActor()
            if actor and self._actor_maps_to_model(actor, self.model_nodes[label]):
                return np.array(picker.GetPickPosition(), dtype=float)
            ds = None
            try:
                ds = picker.GetDataSet()
            except Exception:
                pass
            if ds == self.model_nodes[label].GetPolyData():
                return np.array(picker.GetPickPosition(), dtype=float)
        except Exception:
            return None
        return None

    def _update_osteotomy_from_mouse(self, caller):
        if not self.osteotomy_mouse_mode or not self.active_label:
            return False
        try:
            x, y = caller.GetEventPosition()
            world = self._pick_active_surface_world(x, y)
            if world is None:
                return False
            local = self._world_to_label_local(self.active_label, world)
            self._osteotomy_center_local[self.active_label] = local
            self._update_osteotomy_preview(self.active_label)
            return True
        except Exception as e:
            self._log_event("WARN", f"Mouse broca: {e}")
            return False

    def _refresh_collision_proxy_for_label(self, label):
        visual = self.model_nodes.get(label)
        if not visual or not visual.GetPolyData():
            return
        proxy_poly = self._make_collision_proxy_polydata(visual.GetPolyData())
        if proxy_poly is None:
            return
        node = self.collision_model_nodes.get(label)
        if not node:
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode")
            node.SetName(f"SpineCollisionProxy_{label}")
            node.CreateDefaultDisplayNodes()
            self._add_node_to_scene_folder(node, "proxies")
            self.collision_model_nodes[label] = node
        node.SetAndObservePolyData(proxy_poly)
        if label in self.transform_nodes:
            node.SetAndObserveTransformNodeID(self.transform_nodes[label].GetID())
        dn = node.GetDisplayNode()
        if dn:
            dn.SetVisibility(False)
            self._safe_call(dn, "SetVisibility3D", False)

    def _sync_osteotomy_collision_proxies(self):
        if not self._osteotomy_dirty_labels:
            return False
        dirty = list(self._osteotomy_dirty_labels)
        for label in dirty:
            self._clear_contact_marks_for_label(label)
            self._refresh_collision_proxy_for_label(label)
        self._osteotomy_dirty_labels = set()
        self._last_collision = None
        return True

    def _record_osteotomy_stroke_point(self, label):
        if not label:
            return False
        center = self._osteotomy_center_local.get(label)
        if center is None:
            return False
        center = np.array(center, dtype=float).copy()
        if self._osteotomy_stroke_label != label:
            self._osteotomy_stroke_label = label
            self._osteotomy_stroke_centers = []
        if self._osteotomy_stroke_centers:
            last = np.array(self._osteotomy_stroke_centers[-1], dtype=float)
            min_step = max(0.1, float(self.osteotomy_drill_min_step_mm) * 0.5)
            if float(np.linalg.norm(center - last)) < min_step:
                return False
        self._osteotomy_stroke_centers.append(center)
        return True

    def _apply_osteotomy_stroke_to_label(self, label, centers, finalize=True):
        if not label or label not in self.model_nodes or not centers:
            return 0
        node = self.model_nodes[label]
        poly = node.GetPolyData()
        if poly is None:
            return 0
        if label not in self._osteotomy_original_polydata:
            backup = vtk.vtkPolyData()
            backup.DeepCopy(poly)
            self._osteotomy_original_polydata[label] = backup

        cut_poly, removed = self._clip_polydata_by_local_spheres_fast(poly, centers, self.osteotomy_radius_mm)
        if cut_poly is None or removed <= 0:
            work = poly
            total_removed = 0
            for center in centers:
                next_poly, step_removed = self._clip_polydata_by_local_sphere_fast(work, center, self.osteotomy_radius_mm)
                if next_poly is None or step_removed <= 0:
                    next_poly, step_removed = self._clip_polydata_by_local_sphere(work, center, self.osteotomy_radius_mm)
                if next_poly is not None and step_removed > 0:
                    work = next_poly
                    total_removed += int(step_removed)
            cut_poly = work if total_removed > 0 else None
            removed = total_removed
        if cut_poly is None or removed <= 0:
            return 0

        node.SetAndObservePolyData(cut_poly)
        self._osteotomy_dirty_labels.add(label)
        self._osteotomy_exact_collision_labels.add(label)
        self._osteotomy_cut_count[label] = self._osteotomy_cut_count.get(label, 0) + 1
        self._update_osteotomy_preview(label)
        if finalize:
            self._finalize_osteotomy_drill()
        self._update_status(f"Osteotomia aplicada en {label}: {removed} celdas removidas en {len(centers)} puntos de broca.")
        return int(removed)

    def _commit_osteotomy_stroke(self, finalize=True):
        label = self._osteotomy_stroke_label
        centers = [np.array(c, dtype=float).copy() for c in self._osteotomy_stroke_centers]
        self._osteotomy_stroke_label = None
        self._osteotomy_stroke_centers = []
        if not label or not centers:
            return 0
        removed = self._apply_osteotomy_stroke_to_label(label, centers, finalize=finalize)
        if removed <= 0 and finalize:
            self._update_status(f"Osteotomia {label}: no se eliminaron celdas. Aumenta radio o recoloca broca.")
        return int(removed)

    def _apply_osteotomy_mesh_only(self, label, update_status=False):
        if not label or label not in self.model_nodes:
            return 0
        center = self._osteotomy_center_local.get(label)
        if center is None:
            return 0
        node = self.model_nodes[label]
        poly = node.GetPolyData()
        if poly is None:
            return 0
        if label not in self._osteotomy_original_polydata:
            backup = vtk.vtkPolyData()
            backup.DeepCopy(poly)
            self._osteotomy_original_polydata[label] = backup

        cut_poly, removed = self._clip_polydata_by_local_sphere_fast(poly, center, self.osteotomy_radius_mm)
        if cut_poly is None or removed <= 0:
            cut_poly, removed = self._clip_polydata_by_local_sphere(poly, center, self.osteotomy_radius_mm)
        if cut_poly is None or removed <= 0:
            return 0

        node.SetAndObservePolyData(cut_poly)
        self._osteotomy_dirty_labels.add(label)
        self._osteotomy_exact_collision_labels.add(label)
        self._osteotomy_cut_count[label] = self._osteotomy_cut_count.get(label, 0) + 1
        self._update_osteotomy_preview(label)
        if update_status:
            self._update_status(f"Osteotom?a aplicada en {label}: {removed} celdas removidas.")
        return int(removed)

    def _apply_osteotomy_to_active(self, finalize=True):
        label = self.active_label
        if not label or label not in self.model_nodes:
            self._update_status("Seleccion? una v?rtebra para aplicar osteotom?a.")
            return False
        center = self._osteotomy_center_local.get(label)
        if center is None:
            self._set_osteotomy_center_from_contact()
            center = self._osteotomy_center_local.get(label)
        if center is None:
            return False
        removed = self._apply_osteotomy_mesh_only(label, update_status=False)
        if removed <= 0:
            if finalize:
                self._update_status(f"Osteotom?a {label}: no se eliminaron celdas. Aument? radio o recoloc? broca.")
            return False
        if finalize:
            self._finalize_osteotomy_drill()
        self._update_status(f"Osteotom?a aplicada en {label}: {removed} celdas removidas.")
        return True

    def _should_apply_continuous_drill(self):
        now = time.time()
        if now - float(self._last_drill_time) < float(self.osteotomy_drill_interval_sec):
            return False
        label = self.active_label
        center = self._osteotomy_center_local.get(label) if label else None
        if center is None:
            return False
        if self._last_drill_center_local is not None:
            dist = float(np.linalg.norm(np.array(center, dtype=float) - np.array(self._last_drill_center_local, dtype=float)))
            if dist < float(self.osteotomy_drill_min_step_mm):
                return False
        self._last_drill_time = now
        self._last_drill_center_local = np.array(center, dtype=float).copy()
        return True

    def _begin_osteotomy_drill(self, caller):
        if not self.osteotomy_mouse_mode:
            return False
        if not self._update_osteotomy_from_mouse(caller):
            return False
        self._osteotomy_drilling = True
        self._last_drill_time = 0.0
        self._last_drill_center_local = None
        self._osteotomy_stroke_label = None
        self._osteotomy_stroke_centers = []
        if self.active_label:
            self._clear_contact_marks_for_label(self.active_label)
        self._record_osteotomy_stroke_point(self.active_label)
        if not self.osteotomy_continuous_drill:
            self._osteotomy_drilling = False
            self._commit_osteotomy_stroke(finalize=True)
        return True

    def _continue_osteotomy_drill(self, caller):
        if not (self.osteotomy_mouse_mode and self.osteotomy_continuous_drill and self._osteotomy_drilling):
            return False
        if not self._update_osteotomy_from_mouse(caller):
            return False
        if self._should_apply_continuous_drill():
            self._record_osteotomy_stroke_point(self.active_label)
        return True

    def _finalize_osteotomy_drill(self):
        if not self._osteotomy_dirty_labels:
            return
        self._sync_osteotomy_collision_proxies()
        self._calibrate_collision_baseline()
        self._clear_collision_heatmap()
        self._update_status("Drilling finalizado: proxies recalculados, baseline actualizada y marcas limpiadas.")

    def _end_osteotomy_drill(self):
        if not self._osteotomy_drilling:
            return False
        self._osteotomy_drilling = False
        qt.QTimer.singleShot(0, self._commit_osteotomy_stroke)
        return True

    def _reset_osteotomy_active(self):
        label = self.active_label
        if not label or label not in self._osteotomy_original_polydata:
            self._update_status("La vértebra activa no tiene osteotomía para revertir.")
            return
        restored = vtk.vtkPolyData()
        restored.DeepCopy(self._osteotomy_original_polydata[label])
        self.model_nodes[label].SetAndObservePolyData(restored)
        self._osteotomy_exact_collision_labels.discard(label)
        self._refresh_collision_proxy_for_label(label)
        self._calibrate_collision_baseline()
        self._osteotomy_cut_count[label] = 0
        self._clear_contact_marks_for_label(label)
        self._update_osteotomy_preview(label)
        self._update_status(f"Osteotomía revertida en {label}.")

    # ── Colores ──────────────────────────────────────────────────────────────

    def _set_color(self, label, rgb):
        dn = self.model_nodes[label].GetDisplayNode()
        if dn:
            dn.SetColor(*rgb)

    def _paint_all_normal(self):
        for l in self.ordered_labels:
            if l == self.active_label:
                self._set_color(l, COLOR_SELECTED)
            elif l == self.anchor_label:
                self._set_color(l, COLOR_ANCHOR)
            else:
                self._set_color(l, COLOR_NORMAL)

    # ── Panel Qt ─────────────────────────────────────────────────────────────

    def _build_panel(self):
        self._panel = qt.QWidget()
        self._panel.setWindowTitle(f"SpineSimulator {self.version} - selección 3D + transformada nativa")
        self._panel.setMinimumWidth(390)
        self._panel.resize(430, 760)
        self._panel.setWindowFlags(qt.Qt.Window | qt.Qt.WindowStaysOnTopHint)

        # ------------------------------------------------------------------
        # IMPORTANTE:
        # La interfaz ya tiene muchos controles (dinámica, pivotes,
        # discos, exportación). En pantallas chicas el final quedaba fuera de
        # vista. Por eso todo el panel real vive dentro de un QScrollArea.
        # ------------------------------------------------------------------
        outer = qt.QVBoxLayout(self._panel)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        content = qt.QWidget()
        scroll.setWidget(content)

        root = qt.QVBoxLayout(content)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Selector de vértebra activa y ancla ──
        selBox = qt.QGroupBox("Vértebra activa")
        selLay = qt.QHBoxLayout(selBox)
        self._combo = qt.QComboBox()
        for l in self.ordered_labels:
            self._combo.addItem(l)
        self._combo.currentTextChanged.connect(self._on_vertebra_selected)
        selLay.addWidget(self._combo)
        self._anchor_label_widget = qt.QLabel(f"Ancla actual: {self.anchor_label}")
        self._anchor_label_widget.setStyleSheet("color:#777;font-size:11px")
        selLay.addWidget(self._anchor_label_widget)
        root.addWidget(selBox)

        anchorBox = qt.QGroupBox("Ancla / raíz cinemática")
        anchorLay = qt.QFormLayout(anchorBox)
        self._anchor_combo = qt.QComboBox()
        for l in self.ordered_labels:
            self._anchor_combo.addItem(l)
        if self.anchor_label in self.ordered_labels:
            self._anchor_combo.setCurrentText(self.anchor_label)
        self._anchor_combo.currentTextChanged.connect(self._on_anchor_selected)
        anchorLay.addRow("Vértebra ancla", self._anchor_combo)

        self._anchor_motion_check = qt.QCheckBox("Permitir mover la vértebra ancla")
        self._anchor_motion_check.setChecked(False)
        self._anchor_motion_check.setToolTip("Si está activo, la vértebra ancla también se puede rotar/trasladar. Si está apagado, queda fija como raíz de referencia.")
        self._anchor_motion_check.toggled.connect(self._on_anchor_motion_toggled)
        anchorLay.addRow(self._anchor_motion_check)
        root.addWidget(anchorBox)

        # ── Rotación (control principal) ──
        rotBox = qt.QGroupBox("Rotación (control principal)")
        rotLay = qt.QFormLayout(rotBox)
        self._rot_widgets = {}
        rot_axes = [
            ("Flexión / extensión  (X)", "rx", (-30, 30)),
            ("Rotación axial          (Y)", "ry", (-45, 45)),
            ("Inclinación lateral  (Z)", "rz", (-45, 45)),
        ]
        for name, key, (mn, mx) in rot_axes:
            row = qt.QHBoxLayout()
            sl  = qt.QSlider(qt.Qt.Horizontal)
            sl.setRange(int(mn*10), int(mx*10))
            sl.setValue(0)
            sb  = qt.QDoubleSpinBox()
            sb.setRange(mn, mx)
            sb.setSingleStep(0.5)
            sb.setDecimals(1)
            sb.setSuffix("°")
            sb.setFixedWidth(72)
            sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/10.0))
            sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*10)))
            sl.valueChanged.connect(partial(self._on_rot_slider, key))
            row.addWidget(sl)
            row.addWidget(sb)
            rotLay.addRow(name, row)
            self._rot_widgets[key] = (sl, sb)
        root.addWidget(rotBox)

        # ── Interacción 3D nativa de Slicer ──
        nativeBox = qt.QGroupBox("Interacción 3D nativa de transformada")
        nativeLay = qt.QFormLayout(nativeBox)
        self._native_interaction_check = qt.QCheckBox("Activar handle nativo EN el fiducial/disco de la vértebra seleccionada")
        self._native_interaction_check.setChecked(self.native_transform_interaction_enabled)
        self._native_interaction_check.toggled.connect(self._on_native_interaction_toggled)
        nativeLay.addRow(self._native_interaction_check)

        self._native_rotation_only_check = qt.QCheckBox("Solo rotación en el handle 3D")
        self._native_rotation_only_check.setChecked(self.native_rotation_only)
        self._native_rotation_only_check.toggled.connect(self._on_native_rotation_only_toggled)
        nativeLay.addRow(self._native_rotation_only_check)

        self._native_handle_scale_spin = qt.QDoubleSpinBox()
        self._native_handle_scale_spin.setRange(0.2, 5.0)
        self._native_handle_scale_spin.setSingleStep(0.1)
        self._native_handle_scale_spin.setDecimals(1)
        self._native_handle_scale_spin.setValue(self.native_handle_scale)
        self._native_handle_scale_spin.valueChanged.connect(self._on_native_handle_scale_changed)
        nativeLay.addRow("Tamaño handle", self._native_handle_scale_spin)

        self._native_handle_offset_check = qt.QCheckBox("Desplazar handle hacia la derecha de la pantalla")
        self._native_handle_offset_check.setChecked(self.native_handle_screen_offset_enabled)
        self._native_handle_offset_check.setToolTip("Solo desplaza visualmente el handle para no tapar la columna. La rotación sigue usando el disco/fiducial real como pivot.")
        self._native_handle_offset_check.toggled.connect(self._on_native_handle_offset_toggled)
        nativeLay.addRow(self._native_handle_offset_check)

        self._native_handle_offset_spin = qt.QDoubleSpinBox()
        self._native_handle_offset_spin.setRange(0.0, 250.0)
        self._native_handle_offset_spin.setSingleStep(5.0)
        self._native_handle_offset_spin.setDecimals(1)
        self._native_handle_offset_spin.setSuffix(" mm")
        self._native_handle_offset_spin.setValue(self.native_handle_screen_offset_mm)
        self._native_handle_offset_spin.valueChanged.connect(self._on_native_handle_offset_changed)
        nativeLay.addRow("Offset derecha", self._native_handle_offset_spin)

        hint = qt.QLabel("Uso: clic sobre una vértebra en 3D para seleccionarla. El handle nativo puede mostrarse desplazado a la derecha de la pantalla para no tapar la anatomía; aun así, el cálculo de rotación se aplica alrededor del disco/fiducial real. No se crean aros ni modelos extra.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777;font-size:11px")
        nativeLay.addRow(hint)
        root.addWidget(nativeBox)

        # ── Traslación (ajuste fino, colapsable) ──
        transBox = qt.QGroupBox("Traslación — ajuste fino (mm)")
        transBox.setCheckable(True)
        transBox.setChecked(False)
        transLay = qt.QFormLayout(transBox)
        self._trans_widgets = {}
        trans_axes = [
            ("Lateral  (X)", "tx", (-40, 40)),
            ("Ant/Post (Y)", "ty", (-40, 40)),
            ("Craneal  (Z)", "tz", (-50, 50)),
        ]
        for name, key, (mn, mx) in trans_axes:
            row = qt.QHBoxLayout()
            sl  = qt.QSlider(qt.Qt.Horizontal)
            sl.setRange(int(mn*10), int(mx*10))
            sl.setValue(0)
            sb  = qt.QDoubleSpinBox()
            sb.setRange(mn, mx)
            sb.setSingleStep(0.5)
            sb.setDecimals(1)
            sb.setSuffix(" mm")
            sb.setFixedWidth(72)
            sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/10.0))
            sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*10)))
            sl.valueChanged.connect(partial(self._on_trans_slider, key))
            row.addWidget(sl)
            row.addWidget(sb)
            transLay.addRow(name, row)
            self._trans_widgets[key] = (sl, sb)
        root.addWidget(transBox)

        # Selector de rigidez eliminado: ocupaba pantalla y no era útil para el ajuste de pivotes.

        # ── Dinámica distribuida ──
        dynBox = qt.QGroupBox("Dinámica distribuida")
        dynLay = qt.QFormLayout(dynBox)
        self._dynamic_check = qt.QCheckBox("Mover vecinas al mover una vértebra intermedia")
        self._dynamic_check.setChecked(self.dynamic_enabled)
        self._dynamic_check.toggled.connect(self._on_dynamic_enabled_changed)
        dynLay.addRow(self._dynamic_check)

        self._radius_spin = qt.QSpinBox()
        self._radius_spin.setRange(0, 6)
        self._radius_spin.setValue(int(self.influence_radius))
        self._radius_spin.setSuffix(" niveles")
        self._radius_spin.valueChanged.connect(self._on_influence_radius_changed)
        dynLay.addRow("Alcance", self._radius_spin)

        self._decay_spin = qt.QDoubleSpinBox()
        self._decay_spin.setRange(0.0, 1.0)
        self._decay_spin.setSingleStep(0.05)
        self._decay_spin.setDecimals(2)
        self._decay_spin.setValue(float(self.influence_decay))
        self._decay_spin.valueChanged.connect(self._on_influence_decay_changed)
        dynLay.addRow("Suavidad vecina", self._decay_spin)

        self._chain_check = qt.QCheckBox("Cadena tipo huesos: la vértebra inferior arrastra las superiores")
        self._chain_check.setChecked(self.kinematic_chain_enabled)
        self._chain_check.toggled.connect(self._on_kinematic_chain_changed)
        dynLay.addRow(self._chain_check)

        self._local_bend_spin = qt.QDoubleSpinBox()
        self._local_bend_spin.setRange(0.0, 0.6)
        self._local_bend_spin.setSingleStep(0.05)
        self._local_bend_spin.setDecimals(2)
        self._local_bend_spin.setValue(float(self.local_bend_fraction))
        self._local_bend_spin.valueChanged.connect(self._on_local_bend_changed)
        dynLay.addRow("Curvatura local", self._local_bend_spin)

        root.addWidget(dynBox)

        # ── Colisiones VTP ──
        colBox = qt.QGroupBox("Colisiones VTP")
        colLay = qt.QFormLayout(colBox)
        self._collision_check = qt.QCheckBox("Detectar contacto entre mallas VTP")
        self._collision_check.setChecked(self.collision_enabled)
        self._collision_check.setToolTip("Usa las copias VTP transformadas. Primero filtra por bounds y luego calcula distancia entre superficies.")
        self._collision_check.toggled.connect(self._on_collision_enabled_changed)
        colLay.addRow(self._collision_check)

        self._collision_blocking_check = qt.QCheckBox("Bloquear movimiento al colisionar")
        self._collision_blocking_check.setChecked(self.collision_blocking_enabled)
        self._collision_blocking_check.setToolTip("Apagado por defecto: las vértebras siguen su movimiento natural y solo se pinta la zona de contacto.")
        self._collision_blocking_check.toggled.connect(self._on_collision_blocking_changed)
        colLay.addRow(self._collision_blocking_check)

        self._collision_margin_spin = qt.QDoubleSpinBox()
        self._collision_margin_spin.setRange(0.0, 5.0)
        self._collision_margin_spin.setSingleStep(0.1)
        self._collision_margin_spin.setDecimals(2)
        self._collision_margin_spin.setSuffix(" mm")
        self._collision_margin_spin.setValue(float(self.collision_margin_mm))
        self._collision_margin_spin.valueChanged.connect(self._on_collision_margin_changed)
        colLay.addRow("Margen contacto", self._collision_margin_spin)

        self._collision_radius_spin = qt.QSpinBox()
        self._collision_radius_spin.setRange(1, 8)
        self._collision_radius_spin.setValue(int(self.collision_neighbor_radius))
        self._collision_radius_spin.setSuffix(" arriba/abajo")
        self._collision_radius_spin.valueChanged.connect(self._on_collision_radius_changed)
        colLay.addRow("Vecinas por lado", self._collision_radius_spin)

        self._collision_heatmap_check = qt.QCheckBox("Mostrar contacto visual")
        self._collision_heatmap_check.setChecked(self.collision_heatmap_enabled)
        self._collision_heatmap_check.toggled.connect(self._on_collision_heatmap_changed)
        colLay.addRow(self._collision_heatmap_check)

        self._collision_heatmap_mode_combo = qt.QComboBox()
        self._collision_heatmap_mode_combo.addItem("Superficie roja", "PATCH")
        self._collision_heatmap_mode_combo.addItem("Bolitas rojas", "SPHERES")
        self._collision_heatmap_mode_combo.addItem("Mapa de calor suave", "SURFACE")
        mode = str(self.collision_heatmap_mode).upper()
        mode_index = 2 if mode == "SURFACE" else (1 if mode == "SPHERES" else 0)
        self._collision_heatmap_mode_combo.setCurrentIndex(mode_index)
        self._collision_heatmap_mode_combo.currentIndexChanged.connect(self._on_collision_heatmap_mode_changed)
        colLay.addRow("Visualizacion", self._collision_heatmap_mode_combo)

        self._collision_heatmap_radius_spin = qt.QDoubleSpinBox()
        self._collision_heatmap_radius_spin.setRange(0.5, 15.0)
        self._collision_heatmap_radius_spin.setSingleStep(0.5)
        self._collision_heatmap_radius_spin.setDecimals(1)
        self._collision_heatmap_radius_spin.setSuffix(" mm")
        self._collision_heatmap_radius_spin.setValue(float(self.collision_heatmap_radius_mm))
        self._collision_heatmap_radius_spin.valueChanged.connect(self._on_collision_heatmap_radius_changed)
        colLay.addRow("Radio contacto", self._collision_heatmap_radius_spin)

        self._collision_proxy_check = qt.QCheckBox("Usar proxy liviana para contacto")
        self._collision_proxy_check.setChecked(self.collision_proxy_enabled)
        self._collision_proxy_check.setToolTip("Usa una VTP decimada invisible para contacto; la malla visual se mantiene intacta.")
        self._collision_proxy_check.toggled.connect(self._on_collision_proxy_changed)
        colLay.addRow(self._collision_proxy_check)

        self._collision_sample_spin = qt.QSpinBox()
        self._collision_sample_spin.setRange(100, 3000)
        self._collision_sample_spin.setSingleStep(100)
        self._collision_sample_spin.setValue(int(self.collision_max_sample_points))
        self._collision_sample_spin.setSuffix(" pts")
        self._collision_sample_spin.valueChanged.connect(self._on_collision_sample_points_changed)
        colLay.addRow("Muestreo", self._collision_sample_spin)

        recalibColBtn = qt.QPushButton("Recalibrar postura actual")
        recalibColBtn.setToolTip("Toma los contactos actuales como base permitida. Útil si la malla inicial ya viene rozándose.")
        recalibColBtn.clicked.connect(self._on_collision_recalibrate_clicked)
        colLay.addRow(recalibColBtn)

        clearMarksBtn = qt.QPushButton("Limpiar marcas de contacto")
        clearMarksBtn.setToolTip("Borra las marcas persistentes de contacto sin modificar las vértebras.")
        clearMarksBtn.clicked.connect(self._on_clear_contact_marks_clicked)
        colLay.addRow(clearMarksBtn)
        root.addWidget(colBox)

        # ── Osteotomía virtual ──
        ostBox = qt.QGroupBox("Osteotomía virtual VTP")
        ostLay = qt.QFormLayout(ostBox)

        self._osteotomy_mouse_check = qt.QCheckBox("Modo broca con mouse")
        self._osteotomy_mouse_check.setChecked(self.osteotomy_mouse_mode)
        self._osteotomy_mouse_check.setToolTip("Al activar: mové el mouse sobre la vértebra activa para posicionar la broca; click izquierdo aplica el corte.")
        self._osteotomy_mouse_check.toggled.connect(self._on_osteotomy_mouse_mode_changed)
        ostLay.addRow(self._osteotomy_mouse_check)

        self._osteotomy_continuous_check = qt.QCheckBox("Drill continuo al arrastrar")
        self._osteotomy_continuous_check.setChecked(self.osteotomy_continuous_drill)
        self._osteotomy_continuous_check.setToolTip("V60: mientras se mueve solo acumula la trayectoria; al soltar aplica el corte una sola vez para mayor fluidez.")
        self._osteotomy_continuous_check.toggled.connect(self._on_osteotomy_continuous_changed)
        ostLay.addRow(self._osteotomy_continuous_check)

        self._osteotomy_interval_spin = qt.QDoubleSpinBox()
        self._osteotomy_interval_spin.setRange(0.02, 0.5)
        self._osteotomy_interval_spin.setSingleStep(0.02)
        self._osteotomy_interval_spin.setDecimals(2)
        self._osteotomy_interval_spin.setSuffix(" s")
        self._osteotomy_interval_spin.setValue(float(self.osteotomy_drill_interval_sec))
        self._osteotomy_interval_spin.valueChanged.connect(self._on_osteotomy_interval_changed)
        ostLay.addRow("Intervalo corte", self._osteotomy_interval_spin)

        self._osteotomy_radius_spin = qt.QDoubleSpinBox()
        self._osteotomy_radius_spin.setRange(0.5, 20.0)
        self._osteotomy_radius_spin.setSingleStep(0.5)
        self._osteotomy_radius_spin.setDecimals(1)
        self._osteotomy_radius_spin.setSuffix(" mm")
        self._osteotomy_radius_spin.setValue(float(self.osteotomy_radius_mm))
        self._osteotomy_radius_spin.valueChanged.connect(self._on_osteotomy_radius_changed)
        ostLay.addRow("Radio broca", self._osteotomy_radius_spin)

        placeDrillBtn = qt.QPushButton("Broca en contacto activo")
        placeDrillBtn.setToolTip("Ubica una broca esférica sobre la marca de contacto persistente de la vértebra seleccionada.")
        placeDrillBtn.clicked.connect(self._on_osteotomy_place_from_contact)
        ostLay.addRow(placeDrillBtn)

        applyDrillBtn = qt.QPushButton("Aplicar osteotomía a VTP activo")
        applyDrillBtn.setToolTip("Elimina celdas de la malla VTP activa dentro de la broca. Es un recorte de superficie, no segmentación volumétrica.")
        applyDrillBtn.clicked.connect(self._on_osteotomy_apply)
        ostLay.addRow(applyDrillBtn)

        resetDrillBtn = qt.QPushButton("Revertir osteotomía activa")
        resetDrillBtn.setToolTip("Restaura la malla VTP original de la vértebra activa si ya se aplicó un recorte.")
        resetDrillBtn.clicked.connect(self._on_osteotomy_reset_active)
        ostLay.addRow(resetDrillBtn)

        drillHint = qt.QLabel("Modo mouse: seleccioná una vértebra, activá la broca, mantené click y arrastrá para remover VTP.")
        drillHint.setWordWrap(True)
        drillHint.setStyleSheet("color:#777;font-size:11px")
        ostLay.addRow(drillHint)

        root.addWidget(ostBox)

        # ── Centros / pivotes anatómicos ──
        pivBox = qt.QGroupBox("Pivotes anatómicos")
        pivLay = qt.QFormLayout(pivBox)

        self._pivot_mode_combo = qt.QComboBox()
        self._pivot_mode_combo.addItem("Cuerpo vertebral por densidad (+Y anterior)", "BODY_DENSITY_POS_Y")
        self._pivot_mode_combo.addItem("Cuerpo vertebral por densidad (-Y anterior)", "BODY_DENSITY_NEG_Y")
        self._pivot_mode_combo.addItem("Centro bound completo", "BOUNDS_CENTER")
        self._pivot_mode_combo.addItem("Centro de masa de toda la malla", "CENTER_OF_MASS")
        self._pivot_mode_combo.addItem("Manual: usar fiduciales editables", "MANUAL_FIDUCIALS")
        self._pivot_mode_combo.currentIndexChanged.connect(self._on_pivot_mode_changed)
        pivLay.addRow("Modo pivot", self._pivot_mode_combo)

        self._pivot_check = qt.QCheckBox("Mostrar centros de cuerpos Pivot_XX")
        self._pivot_check.setChecked(self.show_pivot_fiducials)
        self._pivot_check.toggled.connect(self._on_pivot_visibility_changed)
        pivLay.addRow(self._pivot_check)

        self._disc_check = qt.QCheckBox("Mostrar discos Disc_XX_YY celestes (4 mm)")
        self._disc_check.setChecked(self.show_disc_fiducials)
        self._disc_check.toggled.connect(self._on_disc_visibility_changed)
        pivLay.addRow(self._disc_check)

        self._use_disc_check = qt.QCheckBox("Usar discos como pivots de movimiento")
        self._use_disc_check.setChecked(self.use_disc_pivots)
        self._use_disc_check.toggled.connect(self._on_use_disc_pivots_changed)
        pivLay.addRow(self._use_disc_check)

        recalcDiscBtn = qt.QPushButton("Recalcular discos entre cuerpos")
        recalcDiscBtn.setToolTip("Usa la posición actual de Pivot_XX y Pivot_YY para colocar Disc_XX_YY entre ambos cuerpos vertebrales.")
        recalcDiscBtn.clicked.connect(self._on_recalculate_discs_clicked)
        pivLay.addRow(recalcDiscBtn)

        useDiscBtn = qt.QPushButton("Activar discos actuales como pivots")
        useDiscBtn.setToolTip("Lee Disc_XX_YY actuales. Podés moverlos manualmente y después activar este botón.")
        useDiscBtn.clicked.connect(self._on_use_current_discs_clicked)
        pivLay.addRow(useDiscBtn)

        recalcPivotBtn = qt.QPushButton("Recalcular centros de cuerpos automáticos")
        recalcPivotBtn.setToolTip("Recalcula los pivotes con el modo seleccionado y reposiciona los fiduciales amarillos.")
        recalcPivotBtn.clicked.connect(self._on_recalculate_pivots_clicked)
        pivLay.addRow(recalcPivotBtn)

        useManualBtn = qt.QPushButton("Aplicar Pivot_XX manuales")
        useManualBtn.setToolTip("Mové los fiduciales Pivot_XX al cuerpo vertebral. Al activar, el programa los lee automáticamente antes de cada movimiento.")
        useManualBtn.clicked.connect(self._on_use_fiducials_as_pivots_clicked)
        pivLay.addRow(useManualBtn)

        root.addWidget(pivBox)

        # ── Botones ──
        btnRow = qt.QHBoxLayout()
        for txt, fn in [("Reset vértebra", self._on_reset_active),
                         ("Reset todo",    self.reset_all),
                         ("Exportar",      self._on_export)]:
            b = qt.QPushButton(txt)
            b.clicked.connect(fn)
            btnRow.addWidget(b)
        diagBtn = qt.QPushButton("Diagnóstico")
        diagBtn.setToolTip("Verifica nodos, pivotes, discos, transforms y estado general del simulador.")
        diagBtn.clicked.connect(self._on_self_test_clicked)
        btnRow.addWidget(diagBtn)
        root.addLayout(btnRow)

        self._status_lbl = qt.QLabel("Click en un modelo 3D para seleccionar.")
        self._status_lbl.setStyleSheet("color:#777;font-size:11px")
        root.addWidget(self._status_lbl)

        self.active_label = self._combo.currentText
        self._set_color(self.active_label, COLOR_SELECTED)
        self._panel.show()

    # ── Señales del panel ────────────────────────────────────────────────────

    def _on_pivot_visibility_changed(self, checked):
        self._set_pivot_fiducials_visible(bool(checked))
        self._update_status("Centros de cuerpo visibles." if checked else "Centros de cuerpo ocultos.")

    def _on_disc_visibility_changed(self, checked):
        self._set_disc_fiducials_visible(bool(checked))
        self._update_status("Discos intervertebrales visibles." if checked else "Discos intervertebrales ocultos.")

    def _on_use_disc_pivots_changed(self, checked):
        self.use_disc_pivots = bool(checked)
        if checked:
            n = self._read_disc_fiducials_as_motion_pivots(update_status=False)
            self._update_status(f"Usando discos como pivots de movimiento ({n} leídos).")
        else:
            self._update_status("Usando centros de cuerpo Pivot_XX como pivots de movimiento.")
        self._apply_all_transforms()

    def _on_recalculate_discs_clicked(self):
        n = self._recalculate_disc_fiducials_from_body_centers()
        self.use_disc_pivots = True
        if hasattr(self, "_use_disc_check"):
            self._use_disc_check.blockSignals(True)
            self._use_disc_check.setChecked(True)
            self._use_disc_check.blockSignals(False)
        self._apply_all_transforms()
        self._update_status(f"Discos recalculados entre centros de cuerpos: {n}.")

    def _on_use_current_discs_clicked(self):
        n = self._read_disc_fiducials_as_motion_pivots(update_status=False)
        self.use_disc_pivots = True
        self.live_disc_pivots = True
        if hasattr(self, "_use_disc_check"):
            self._use_disc_check.blockSignals(True)
            self._use_disc_check.setChecked(True)
            self._use_disc_check.blockSignals(False)
        self._apply_all_transforms()
        self._update_status(f"Modo disco LIVE: usando Disc_XX_YY actuales como pivots ({n}).")

    def _on_pivot_mode_changed(self, index):
        try:
            mode = self._pivot_mode_combo.itemData(index)
        except Exception:
            mode = None
        if mode:
            self.pivot_mode = str(mode)
        if self.pivot_mode == "MANUAL_FIDUCIALS":
            self.manual_pivots_enabled = True
            self.live_fiducial_pivots = True
            self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)
            self._update_status("Modo manual LIVE: mové Pivot_XX; el próximo movimiento los toma automáticamente.")
        else:
            self._recompute_automatic_pivots()
            self._update_status(f"Pivotes recalculados con modo: {self.pivot_mode}")

    def _on_recalculate_pivots_clicked(self):
        if self.pivot_mode == "MANUAL_FIDUCIALS":
            self._use_current_fiducials_as_pivots()
        else:
            self._recompute_automatic_pivots()
            self._update_status(f"Pivotes automáticos recalculados: {self.pivot_mode}")

    def _on_use_fiducials_as_pivots_clicked(self):
        updated = self._refresh_base_positions_from_fiducials(rebuild_solver=True, update_status=False)
        self.manual_pivots_enabled = True
        self.live_fiducial_pivots = True
        self.pivot_mode = "MANUAL_FIDUCIALS"
        self.use_disc_pivots = False
        self.live_disc_pivots = False
        self._motion_pivots = {}
        try:
            if hasattr(self, "_use_disc_check") and self._use_disc_check:
                self._use_disc_check.blockSignals(True)
                self._use_disc_check.setChecked(False)
                self._use_disc_check.blockSignals(False)
        except Exception:
            pass
        self._apply_all_transforms()
        self._update_native_transform_interaction()
        self._update_status(f"Pivots manuales activos: Pivot_XX es el pivot real ({updated} leidos). Discos desactivados como pivots.")


    def _on_anchor_selected(self, label):
        if not label or label not in self.ordered_labels:
            return
        old_anchor = self.anchor_label
        self.anchor_label = label
        # Reconstruye el solver con el nuevo índice de ancla sin recalcular pivots.
        self._build_solver()
        if not self.anchor_motion_enabled:
            self._rotations[label] = [0.0, 0.0, 0.0]
            self._translations[label] = [0.0, 0.0, 0.0]
        self._apply_all_transforms()
        self._paint_all_normal()
        if self._anchor_label_widget:
            self._anchor_label_widget.setText(f"Ancla actual: {self.anchor_label}")
        self._update_status(f"Ancla cambiada: {old_anchor} → {self.anchor_label}. Colisiones por defecto: apagadas.")

    def _on_anchor_motion_toggled(self, checked):
        self.anchor_motion_enabled = bool(checked)
        if self.anchor_motion_enabled:
            self._update_status(f"Ancla desbloqueada: {self.anchor_label} ahora puede editarse.")
        else:
            # Al volver a bloquear, la vértebra ancla queda fija/identidad.
            if self.anchor_label in self._rotations:
                self._rotations[self.anchor_label] = [0.0, 0.0, 0.0]
                self._translations[self.anchor_label] = [0.0, 0.0, 0.0]
            self._apply_all_transforms()
            self._sync_rot_sliders()
            self._sync_trans_sliders()
            self._update_status(f"Ancla bloqueada: {self.anchor_label} queda fija como raíz cinemática.")
        self._paint_all_normal()

    def _on_vertebra_selected(self, label):
        prev = self.active_label
        if prev:
            if prev == self.anchor_label:
                self._set_color(prev, COLOR_ANCHOR)
            else:
                self._set_color(prev, COLOR_NORMAL)
        self.active_label = label
        self._set_color(label, COLOR_SELECTED)
        self._reset_widgets()
        p = self._get_current_pivot_world(label)
        mode_txt = "disco inferior" if self.use_disc_pivots and label in self._motion_pivots else "centro cuerpo"
        self._update_native_transform_interaction()
        if self._osteotomy_preview_node:
            self._update_osteotomy_preview(label)
        self._update_status(f"Activa: {label} | región: {get_region(label)} | pivot movimiento: {mode_txt} | centro cuerpo RAS: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    def _on_rot_slider(self, axis, raw):
        if not self.active_label: return
        rx = self._rot_widgets["rx"][0].value / 10.0
        ry = self._rot_widgets["ry"][0].value / 10.0
        rz = self._rot_widgets["rz"][0].value / 10.0
        label = self.active_label
        # Calcular deltas respecto al estado acumulado
        drx = rx - self._rotations[label][0]
        dry = ry - self._rotations[label][1]
        drz = rz - self._rotations[label][2]
        self.apply_rotation(label, drx, dry, drz)
        # Si fue revertido, resincronizar sliders con el estado real
        self._sync_rot_sliders()

    def _on_trans_slider(self, axis, raw):
        if not self.active_label: return
        tx = self._trans_widgets["tx"][0].value / 10.0
        ty = self._trans_widgets["ty"][0].value / 10.0
        tz = self._trans_widgets["tz"][0].value / 10.0
        self.apply_translation(self.active_label, tx, ty, tz)
        self._sync_trans_sliders()

    def _on_rigidity_changed(self, region, value):
        REGION_STIFFNESS[region] = value / 100.0
        for i, l in enumerate(self.ordered_labels):
            if get_region(l) == region:
                self.solver.stiff[i] = value / 100.0

    def _on_dynamic_enabled_changed(self, checked):
        self.dynamic_enabled = bool(checked)
        self._update_status("Dinámica distribuida activada" if checked else "Dinámica distribuida desactivada")

    def _on_influence_radius_changed(self, value):
        self.influence_radius = int(value)
        self._update_status(f"Alcance dinámico: {self.influence_radius} niveles")

    def _on_influence_decay_changed(self, value):
        self.influence_decay = float(value)
        self._update_status(f"Peso vecina: {self.influence_decay:.2f}")

    def _on_kinematic_chain_changed(self, checked):
        self.kinematic_chain_enabled = bool(checked)
        modo = "cadena craneal tipo huesos" if self.kinematic_chain_enabled else "rotación local/distribuida clásica"
        self._apply_all_transforms()
        self._update_status(f"Modo de movimiento: {modo}")

    def _on_local_bend_changed(self, value):
        self.local_bend_fraction = float(value)
        self._apply_all_transforms()
        self._update_status(f"Curvatura local: {self.local_bend_fraction:.2f}")

    def _on_collision_enabled_changed(self, checked):
        self.collision_enabled = bool(checked)
        if self.collision_enabled:
            self._calibrate_collision_baseline()
            self._update_status("Contacto VTP activado. Bloqueo activo si la casilla de bloqueo esta marcada.")
        else:
            self._clear_collision_heatmap()
            self._update_status("Contacto VTP desactivado.")

    def _on_collision_blocking_changed(self, checked):
        self.collision_blocking_enabled = bool(checked)
        self._calibrate_collision_baseline()
        self._update_status("Bloqueo por colisión activado." if checked else "Bloqueo desactivado: movimiento libre con mapa de contacto.")

    def _on_collision_margin_changed(self, value):
        self.collision_margin_mm = float(value)
        self._calibrate_collision_baseline()
        self._update_status(f"Margen de colisión: {self.collision_margin_mm:.2f} mm")

    def _on_collision_radius_changed(self, value):
        self.collision_neighbor_radius = int(value)
        self._calibrate_collision_baseline()
        self._update_status(f"Contacto VTP: revisando {self.collision_neighbor_radius} arriba y {self.collision_neighbor_radius} abajo")

    def _on_collision_heatmap_changed(self, checked):
        self.collision_heatmap_enabled = bool(checked)
        if not self.collision_heatmap_enabled:
            self._clear_collision_heatmap()
        self._update_status("Contacto visual activado." if checked else "Contacto visual desactivado.")

    def _on_collision_heatmap_mode_changed(self, index):
        try:
            mode = self._collision_heatmap_mode_combo.itemData(index)
        except Exception:
            mode = "PATCH"
        self.collision_heatmap_mode = str(mode or "PATCH")
        self._clear_collision_heatmap()
        txt = "mapa de calor suave" if self.collision_heatmap_mode == "SURFACE" else ("bolitas rojas" if self.collision_heatmap_mode == "SPHERES" else "superficie roja")
        self._update_status(f"Visualizacion de contacto: {txt}.")

    def _on_collision_heatmap_radius_changed(self, value):
        self.collision_heatmap_radius_mm = float(value)
        self._update_status(f"Radio visual de contacto: {self.collision_heatmap_radius_mm:.1f} mm")

    def _on_collision_proxy_changed(self, checked):
        self.collision_proxy_enabled = bool(checked)
        if self.collision_proxy_enabled and not self.collision_model_nodes:
            self._create_collision_proxy_nodes()
            for label, node in self.collision_model_nodes.items():
                if label in self.transform_nodes:
                    node.SetAndObserveTransformNodeID(self.transform_nodes[label].GetID())
        self._calibrate_collision_baseline()
        self._update_status("Proxy liviana de contacto activada." if checked else "Contacto usando malla visual completa.")

    def _on_collision_sample_points_changed(self, value):
        self.collision_max_sample_points = int(value)
        self._update_status(f"Muestreo de contacto: {self.collision_max_sample_points} puntos por malla")

    def _on_collision_recalibrate_clicked(self):
        self._calibrate_collision_baseline()
        n = len(self._collision_baseline_pairs)
        self._update_status(f"Colisiones recalibradas. Contactos actuales permitidos: {n} pares.")

    def _on_clear_contact_marks_clicked(self):
        self._clear_collision_heatmap()
        self._update_status("Marcas de contacto limpiadas.")

    def _on_osteotomy_radius_changed(self, value):
        self.osteotomy_radius_mm = float(value)
        if self.active_label:
            self._update_osteotomy_preview(self.active_label)
        self._update_status(f"Radio de broca: {self.osteotomy_radius_mm:.1f} mm")

    def _on_osteotomy_mouse_mode_changed(self, checked):
        self.osteotomy_mouse_mode = bool(checked)
        if self.osteotomy_mouse_mode:
            self.native_transform_interaction_enabled = False
            if hasattr(self, "_native_interaction_check"):
                self._native_interaction_check.blockSignals(True)
                self._native_interaction_check.setChecked(False)
                self._native_interaction_check.blockSignals(False)
            self._disable_all_native_transform_interactions()
            if self.active_label:
                self._update_osteotomy_preview(self.active_label)
            self._update_status("Modo broca activo: mantene ESPACIO y move el mouse sobre la vertebra activa.")
        else:
            self._end_osteotomy_drill()
            self._remove_osteotomy_preview()
            self._update_status("Modo broca desactivado. Click 3D vuelve a seleccionar vertebras.")

    def _on_osteotomy_continuous_changed(self, checked):
        self.osteotomy_continuous_drill = bool(checked)
        self._update_status("Drill continuo activado." if checked else "Drill continuo desactivado: click corta una vez.")

    def _on_osteotomy_interval_changed(self, value):
        self.osteotomy_drill_interval_sec = float(value)
        self._update_status(f"Intervalo de corte continuo: {self.osteotomy_drill_interval_sec:.2f} s")

    def _on_osteotomy_place_from_contact(self):
        self._set_osteotomy_center_from_contact()

    def _on_osteotomy_apply(self):
        self._apply_osteotomy_to_active()

    def _on_osteotomy_reset_active(self):
        self._reset_osteotomy_active()

    def _on_reset_active(self):
        if self.active_label:
            self.reset_vertebra(self.active_label)

    def _on_export(self):
        import os, json
        from datetime import datetime
        folder = qt.QFileDialog.getExistingDirectory(self._panel, "Carpeta de exportación")
        if not folder: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "version": self.version,
            "timestamp": ts,
            "anchor": self.anchor_label,
            "active_label": self.active_label,
            "pivot_mode": self.pivot_mode,
            "use_disc_pivots": bool(self.use_disc_pivots),
            "kinematic_chain_enabled": bool(self.kinematic_chain_enabled),
            "influence_radius": int(self.influence_radius),
            "influence_decay": float(self.influence_decay),
            "local_bend_fraction": float(self.local_bend_fraction),
            "collision_enabled": bool(self.collision_enabled),
            "collision_blocking_enabled": bool(self.collision_blocking_enabled),
            "collision_margin_mm": float(self.collision_margin_mm),
            "collision_neighbor_radius": int(self.collision_neighbor_radius),
            "collision_neighbors_each_side": int(self.collision_neighbor_radius),
            "collision_heatmap_enabled": bool(self.collision_heatmap_enabled),
            "collision_heatmap_mode": str(self.collision_heatmap_mode),
            "collision_heatmap_radius_mm": float(self.collision_heatmap_radius_mm),
            "collision_proxy_enabled": bool(self.collision_proxy_enabled),
            "collision_proxy_reduction": float(self.collision_proxy_reduction),
            "collision_max_sample_points": int(self.collision_max_sample_points),
            "collision_heatmap_max_points": int(self.collision_heatmap_max_points),
            "contact_marks_persistent": bool(self.contact_marks_persistent),
            "contact_marks_follow_vertebra": bool(self.contact_marks_follow_vertebra),
            "contact_mark_pairs": [list(k) for k in sorted(self._contact_mark_nodes_by_pair.keys())],
            "osteotomy_radius_mm": float(self.osteotomy_radius_mm),
            "osteotomy_mouse_mode": bool(self.osteotomy_mouse_mode),
            "osteotomy_continuous_drill": bool(self.osteotomy_continuous_drill),
            "osteotomy_drill_interval_sec": float(self.osteotomy_drill_interval_sec),
            "osteotomy_drill_min_step_mm": float(self.osteotomy_drill_min_step_mm),
            "osteotomy_cut_count": dict(self._osteotomy_cut_count),
            "osteotomy_labels_with_backup": sorted(self._osteotomy_original_polydata.keys()),
            "collision_baseline_pairs": [
                {
                    "pair": list(p),
                    "min_abs_mm": round(float(v.get("min_abs_mm", 0.0)), 4),
                    "min_signed_mm": round(float(v.get("min_signed_mm", 0.0)), 4),
                }
                for p, v in sorted(self._collision_baseline_pairs.items())
            ],
            "vtp_output_dir": self.vtp_output_dir,
            "vertebrae": {},
        }
        csv_rows = []
        for l in self.ordered_labels:
            t = self.transform_nodes[l]
            slicer.util.saveNode(t, os.path.join(folder, f"SpineV3Center_{l}_{ts}.tfm"))
            m = vtk.vtkMatrix4x4()
            t.GetMatrixTransformToParent(m)
            matrix = [[round(float(m.GetElement(r, c)), 6) for c in range(4)] for r in range(4)]
            pivot = self._base_positions.get(l)
            motion_pivot = self._motion_pivots.get(l)
            rx, ry, rz = self._rotations[l]
            tx, ty, tz = self._translations[l]
            item = {
                "rx": round(self._rotations[l][0], 2),
                "ry": round(self._rotations[l][1], 2),
                "rz": round(self._rotations[l][2], 2),
                "tx_slider": round(tx, 2),
                "ty_slider": round(ty, 2),
                "tz_slider": round(tz, 2),
                "matrix_tx": round(m.GetElement(0,3), 2),
                "matrix_ty": round(m.GetElement(1,3), 2),
                "matrix_tz": round(m.GetElement(2,3), 2),
                "base_pivot": [round(float(x), 3) for x in pivot] if pivot is not None else None,
                "motion_pivot": [round(float(x), 3) for x in motion_pivot] if motion_pivot is not None else None,
                "matrix": matrix,
                "vtp_path": self.converted_vtp_paths.get(l),
            }
            summary["vertebrae"][l] = item
            csv_rows.append({
                "label": l,
                "rx_deg": round(rx, 3),
                "ry_deg": round(ry, 3),
                "rz_deg": round(rz, 3),
                "tx_slider_mm": round(tx, 3),
                "ty_slider_mm": round(ty, 3),
                "tz_slider_mm": round(tz, 3),
                "matrix_tx_mm": round(m.GetElement(0,3), 3),
                "matrix_ty_mm": round(m.GetElement(1,3), 3),
                "matrix_tz_mm": round(m.GetElement(2,3), 3),
                "anchor": int(l == self.anchor_label),
                "vtp_path": self.converted_vtp_paths.get(l, ""),
            })
        json_path = os.path.join(folder, f"SpineSimulator_{self.version}_{ts}.json")
        csv_path = os.path.join(folder, f"SpineSimulator_{self.version}_{ts}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
            if csv_rows:
                writer.writeheader()
                writer.writerows(csv_rows)
        self._update_status(f"Exportado TFM + JSON + CSV en: {folder}")

    def _on_self_test_clicked(self):
        report = self.self_test(verbose=True)
        status = "OK" if report["ok"] else "Revisar"
        self._update_status(f"Diagnóstico {status}: {len(report['issues'])} problemas, {len(report['warnings'])} avisos.")

    def self_test(self, verbose=True):
        """Diagnóstico rápido del estado interno y de los nodos MRML creados."""
        report = {"ok": True, "issues": [], "warnings": [], "counts": {}}
        labels = list(self.ordered_labels)
        report["counts"]["vertebrae"] = len(labels)
        if not labels:
            report["issues"].append("No hay vértebras detectadas. Ejecutá sim.start().")
        missing_models = [l for l in labels if l not in self.model_nodes or self.model_nodes[l] is None]
        missing_transforms = [l for l in labels if l not in self.transform_nodes or self.transform_nodes[l] is None]
        missing_pivots = [l for l in labels if l not in self._base_positions]
        if missing_models:
            report["issues"].append("Faltan modelos VTP: " + ", ".join(missing_models))
        if missing_transforms:
            report["issues"].append("Faltan transforms: " + ", ".join(missing_transforms))
        if missing_pivots:
            report["issues"].append("Faltan pivotes base: " + ", ".join(missing_pivots))
        if self.anchor_label not in labels:
            report["issues"].append(f"Ancla inválida: {self.anchor_label}")
        if self.solver is None:
            report["issues"].append("Solver FABRIK no inicializado.")
        elif len(getattr(self.solver, "pos", [])) != len(labels):
            report["issues"].append("Solver con cantidad de posiciones distinta a la cadena.")
        if self.use_disc_pivots:
            expected_discs = max(0, len(labels) - 1)
            actual_discs = len(self._disc_point_indices)
            report["counts"]["discs"] = actual_discs
            if actual_discs < expected_discs:
                report["warnings"].append(f"Discos incompletos: {actual_discs}/{expected_discs}.")
        report["counts"]["collision_baseline_pairs"] = len(self._collision_baseline_pairs)
        report["counts"]["collision_heat_labels"] = len(self._collision_heat_labels)
        report["counts"]["collision_overlay_nodes"] = len(self._collision_overlay_nodes)
        report["counts"]["contact_mark_pairs"] = len(self._contact_mark_nodes_by_pair)
        report["counts"]["collision_blocking_enabled"] = int(bool(self.collision_blocking_enabled))
        report["counts"]["collision_proxy_nodes"] = len(self.collision_model_nodes)
        report["counts"]["osteotomy_labels"] = len([l for l, n in self._osteotomy_cut_count.items() if n > 0])
        if self.collision_enabled and self.collision_margin_mm < 0.05:
            report["warnings"].append("Contacto VTP activado con margen muy bajo; puede no pintar contactos leves.")
        if self.collision_enabled and self.collision_neighbor_radius < 1:
            report["warnings"].append("Radio de colisión menor a 1; no se revisarán vecinas.")
        if self.native_transform_interaction_enabled and self.active_label and not self._interaction_handle_node:
            report["warnings"].append("Handle nativo activo pero el nodo de handle todavía no existe.")
        if len(self._event_log) > 0:
            report["counts"]["log_events"] = len(self._event_log)
        report["ok"] = len(report["issues"]) == 0
        if verbose:
            print(f"\n[SpineSimulator {self.version}] Diagnóstico")
            print("  Estado:", "OK" if report["ok"] else "REVISAR")
            for issue in report["issues"]:
                print("  [PROBLEMA]", issue)
            for warning in report["warnings"]:
                print("  [AVISO]", warning)
            if not report["issues"] and not report["warnings"]:
                print("  Sin problemas detectados.")
        return report

    # ── Sincronización de widgets ────────────────────────────────────────────

    def _reset_widgets(self):
        for widgets in [self._rot_widgets, self._trans_widgets]:
            for sl, sb in widgets.values():
                sl.blockSignals(True); sb.blockSignals(True)
                sl.setValue(0);        sb.setValue(0.0)
                sl.blockSignals(False); sb.blockSignals(False)
        if self.active_label:
            self._sync_rot_sliders()
            self._sync_trans_sliders()

    def _sync_rot_sliders(self):
        if not self.active_label: return
        rx, ry, rz = self._rotations[self.active_label]
        for key, val in [("rx",rx),("ry",ry),("rz",rz)]:
            sl, sb = self._rot_widgets[key]
            sl.blockSignals(True); sb.blockSignals(True)
            sl.setValue(int(val*10)); sb.setValue(val)
            sl.blockSignals(False); sb.blockSignals(False)

    def _sync_trans_sliders(self):
        if not self.active_label: return
        tx, ty, tz = self._translations[self.active_label]
        for key, val in [("tx",tx),("ty",ty),("tz",tz)]:
            sl, sb = self._trans_widgets[key]
            sl.blockSignals(True); sb.blockSignals(True)
            sl.setValue(int(val*10)); sb.setValue(val)
            sl.blockSignals(False); sb.blockSignals(False)

    def _update_status(self, msg):
        if self._status_lbl:
            self._status_lbl.setText(msg)

    def _log_event(self, level, msg):
        item = {
            "time": time.strftime("%H:%M:%S"),
            "level": str(level).upper(),
            "message": str(msg),
        }
        self._event_log.append(item)
        if len(self._event_log) > 250:
            self._event_log = self._event_log[-250:]
        if self.debug_enabled or item["level"] in ("ERROR", "WARN"):
            print(f"[SpineSimulator {self.version} {item['level']}] {item['message']}")


    # ── Interacción nativa de transformada de Slicer ───────────────────────────

    def _safe_call(self, obj, method_names, *args):
        """Llama un método si existe. Evita romper entre versiones de Slicer."""
        if obj is None:
            return False
        if isinstance(method_names, str):
            method_names = [method_names]
        for name in method_names:
            try:
                if hasattr(obj, name):
                    getattr(obj, name)(*args)
                    return True
            except Exception:
                pass
        return False

    def _motion_pivot_world(self, label):
        """Pivot/disco activo en coordenadas RAS/world actuales."""
        if not label or label not in self.transform_nodes:
            return None
        local = self._get_motion_pivot_local(label)
        m = vtk.vtkMatrix4x4()
        self.transform_nodes[label].GetMatrixTransformToWorld(m)
        inp = [float(local[0]), float(local[1]), float(local[2]), 1.0]
        out = [0.0, 0.0, 0.0, 0.0]
        m.MultiplyPoint(inp, out)
        return np.array(out[:3], dtype=float)

    def _vtk_matrix_to_np(self, m):
        arr = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                arr[r, c] = float(m.GetElement(r, c))
        return arr

    def _np_to_vtk_matrix(self, arr):
        m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(arr[r, c]))
        return m

    def _euler_xyz_from_rotation_matrix(self, R):
        """Inversa aproximada de R = Rz @ Ry @ Rx, devuelve grados."""
        R = np.array(R, dtype=float)
        sy = -R[2, 0]
        sy = max(-1.0, min(1.0, float(sy)))
        ry = np.arcsin(sy)
        cy = np.cos(ry)
        if abs(cy) > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = 0.0
            rz = np.arctan2(-R[0, 1], R[1, 1])
        return [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]

    def _make_models_pickable(self):
        """Asegura que las copias VTP puedan seleccionarse desde la vista 3D."""
        for label, node in self.model_nodes.items():
            try:
                node.SetSelectable(1)
            except Exception:
                pass
            try:
                node.CreateDefaultDisplayNodes()
                dn = node.GetDisplayNode()
                if dn:
                    dn.SetVisibility(True)
                    self._safe_call(dn, "SetSelectable", True)
                    self._safe_call(dn, "SetPickable", True)
                    # Evita que los VTP queden detrás de los originales invisibles o display no pickeable.
                    self._safe_call(dn, "SetVisibility3D", True)
            except Exception:
                pass

    def _disable_all_native_transform_interactions(self):
        """Oculta el handle nativo único y desactiva handles de transforms vertebrales."""
        # Ocultar transforms propios de cada vértebra, porque el handle real es el único nodo SpineV3Handle_Active.
        for label, tnode in list(self.transform_nodes.items()):
            try:
                tnode.CreateDefaultDisplayNodes()
                dn = tnode.GetDisplayNode()
                if dn:
                    dn.SetEditorVisibility(False)
                    dn.SetVisibility(False)
            except Exception:
                pass
        try:
            if self._interaction_handle_node:
                self._interaction_handle_node.CreateDefaultDisplayNodes()
                dn = self._interaction_handle_node.GetDisplayNode()
                if dn:
                    dn.SetEditorVisibility(False)
                    dn.SetVisibility(False)
        except Exception:
            pass

    def _ensure_interaction_handle_node(self):
        """Crea un transform nativo único que Slicer puede mostrar como handle.

        Este nodo no se aplica directamente a la malla. Solo sirve como controlador visual.
        Cuando el usuario lo rota, el observer copia esa rotación a self._rotations[label]
        y el simulador reconstruye la matriz final alrededor del disco/pivot real.
        """
        if self._interaction_handle_node and self.scene.GetNodeByID(self._interaction_handle_node.GetID()):
            return self._interaction_handle_node
        node = self.scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "SpineV3Handle_ActivePivot")
        node.CreateDefaultDisplayNodes()
        self._add_node_to_scene_folder(node, "transforms")
        self._interaction_handle_node = node
        self._interaction_handle_observer = node.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent,
            self._on_interaction_handle_modified
        )
        return node

    def _configure_transform_display_node(self, dn, label):
        """Configura el display nativo del handle.

        En Slicer 5.8+ los métodos están bajo SetEditor*.
        """
        if not dn:
            return
        dn.SetVisibility(True)
        dn.SetEditorVisibility(True)
        dn.SetEditorRotationEnabled(True)
        dn.SetEditorTranslationEnabled(not bool(self.native_rotation_only))
        dn.SetEditorScalingEnabled(False)

    def _camera_right_vector_world(self):
        """Devuelve el vector 'derecha de pantalla' en coordenadas RAS/world.

        Se usa solo para correr visualmente el handle nativo y que no tape la
        columna. Si no se puede leer la cámara, vuelve a +X como respaldo.
        """
        try:
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
            renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
            cam = renderer.GetActiveCamera()
            dop = np.array(cam.GetDirectionOfProjection(), dtype=float)
            up = np.array(cam.GetViewUp(), dtype=float)
            right = np.cross(dop, up)
            n = np.linalg.norm(right)
            if n < 1e-6:
                return np.array([1.0, 0.0, 0.0], dtype=float)
            return right / n
        except Exception:
            return np.array([1.0, 0.0, 0.0], dtype=float)

    def _interaction_handle_display_position(self, pivot_world):
        """Posición visual del handle.

        El pivot real de movimiento sigue siendo pivot_world. El offset solo
        mueve el widget nativo a la derecha de pantalla para despejar la visión.
        """
        p = np.array(pivot_world, dtype=float)
        if getattr(self, "native_handle_screen_offset_enabled", True):
            p = p + self._camera_right_vector_world() * float(getattr(self, "native_handle_screen_offset_mm", 65.0))
        return p

    def _sync_interaction_handle_to_active(self):
        """Coloca el handle nativo cerca del pivot/disco activo.

        En v3.17 el handle puede verse desplazado a la derecha de la pantalla
        para no tapar la anatomía. La rotación que el usuario hace en ese
        handle se aplica igualmente alrededor del disco/fiducial real.
        """
        if not self.active_label or self.active_label not in self.transform_nodes:
            return
        hnode = self._ensure_interaction_handle_node()
        pivot_world = self._motion_pivot_world(self.active_label)
        if pivot_world is None:
            return

        handle_pos = self._interaction_handle_display_position(pivot_world)
        rx, ry, rz = self._rotations.get(self.active_label, [0.0, 0.0, 0.0])
        R = self._np_rotation_xyz(rx, ry, rz)
        M = np.eye(4, dtype=float)
        M[:3, :3] = R[:3, :3]
        M[:3, 3] = np.array(handle_pos, dtype=float)

        self._interaction_handle_updating = True
        try:
            hnode.SetMatrixTransformToParent(self._np_to_vtk_matrix(M))
            hnode.CreateDefaultDisplayNodes()
            dn = hnode.GetDisplayNode()
            self._configure_transform_display_node(dn, self.active_label)
            self._interaction_handle_active_label = self.active_label
            try:
                hnode.SetName(f"SpineV3Handle_{self.active_label}_offsetDerecha")
            except Exception:
                pass
        finally:
            self._interaction_handle_updating = False

    def _update_native_transform_interaction(self):
        """Muestra un único handle nativo ubicado en el fiducial/disco activo."""
        if not getattr(self, "native_transform_interaction_enabled", True):
            self._disable_all_native_transform_interactions()
            return
        # Apagar cualquier handle accidental de los transforms de las vértebras.
        for label, tnode in list(self.transform_nodes.items()):
            try:
                tnode.CreateDefaultDisplayNodes()
                dn = tnode.GetDisplayNode()
                if dn:
                    self._safe_call(dn, ["SetEditorVisibility", "SetInteractionVisible", "SetHandlesInteractive"], False)
                    self._safe_call(dn, "SetVisibility", False)
            except Exception:
                pass
        self._sync_interaction_handle_to_active()

    def _on_interaction_handle_modified(self, caller, event=None):
        """Aplica rotación y traslación del handle nativo al modelo activo."""
        if self._interaction_handle_updating:
            return
        label = self._interaction_handle_active_label or self.active_label
        if not label or label not in self._rotations:
            return
        try:
            m = vtk.vtkMatrix4x4()
            caller.GetMatrixTransformToParent(m)
            M = self._vtk_matrix_to_np(m)
            rx, ry, rz = self._euler_xyz_from_rotation_matrix(M[:3, :3])

            rx = max(-30.0, min(30.0, rx))
            ry = max(-45.0, min(45.0, ry))
            rz = max(-45.0, min(45.0, rz))

            # Traslación: delta desde la posición de referencia del handle
            if not self.native_rotation_only:
                pivot_w = self._motion_pivot_world(label)
                if pivot_w is None:
                    pivot_w = [0.0, 0.0, 0.0]
                ref_pos = self._interaction_handle_display_position(pivot_w)
                now_pos = [float(M[0, 3]), float(M[1, 3]), float(M[2, 3])]
                dx = float(now_pos[0]) - float(ref_pos[0])
                dy = float(now_pos[1]) - float(ref_pos[1])
                dz = float(now_pos[2]) - float(ref_pos[2])

                prev_tx, prev_ty, prev_tz = [float(x) for x in self._translations.get(label, [0.0, 0.0, 0.0])]
                self._translations[label] = [
                    max(-30.0, min(30.0, prev_tx + dx)),
                    max(-30.0, min(30.0, prev_ty + dy)),
                    max(-30.0, min(30.0, prev_tz + dz))
                ]

            self._save_snapshot()
            self._rotations[label] = [rx, ry, rz]
            self._apply_all_transforms()
            if self._reject_move_if_collision([label]):
                return
            self._sync_rot_sliders()
            self._sync_trans_sliders()
            if not self._last_collision:
                mode = "rot+trasl" if not self.native_rotation_only else "rotación"
                self._update_status(f"Handle 3D: {label} {mode}")
        except Exception as e:
            print(f"[Handle 3D] Error: {e}")

    def _on_native_interaction_toggled(self, checked):
        self.native_transform_interaction_enabled = bool(checked)
        self._update_native_transform_interaction()
        self._update_status("Handle nativo de transformada activado." if checked else "Handle nativo de transformada desactivado.")

    def _on_native_rotation_only_toggled(self, checked):
        self.native_rotation_only = bool(checked)
        self._update_native_transform_interaction()

    def _on_native_handle_scale_changed(self, value):
        self.native_handle_scale = float(value)
        self._update_native_transform_interaction()

    def _on_native_handle_offset_toggled(self, checked):
        self.native_handle_screen_offset_enabled = bool(checked)
        self._update_native_transform_interaction()
        self._update_status("Offset visual del handle activado." if checked else "Offset visual del handle desactivado: el handle vuelve al disco/pivot.")

    def _on_native_handle_offset_changed(self, value):
        self.native_handle_screen_offset_mm = float(value)
        self._update_native_transform_interaction()

    def _cleanup_rotation_gizmo(self):
        """Compatibilidad con versiones anteriores: ya no se crean aros/modelos."""
        return

    def _on_3d_mouse_move(self, caller, event):
        if self.osteotomy_mouse_mode:
            self._update_osteotomy_from_mouse(caller)
            self._continue_osteotomy_drill(caller)
        return

    def _on_3d_left_release(self, caller, event):
        if self.osteotomy_mouse_mode:
            self._end_osteotomy_drill()
        return

    def _display_manager_pick_label(self, x, y):
        """Intenta usar el displayable manager de Slicer para identificar el MRML node."""
        try:
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
            dm = view.displayableManagerByClassName("vtkMRMLModelDisplayableManager")
            if dm and hasattr(dm, "Pick"):
                picked = dm.Pick(int(x), int(y))
                if picked:
                    # Pick puede devolver model node, display node u otro MRML node según versión.
                    ids = []
                    try: ids.append(picked.GetID())
                    except Exception: pass
                    try:
                        if hasattr(picked, "GetDisplayableNode") and picked.GetDisplayableNode():
                            ids.append(picked.GetDisplayableNode().GetID())
                    except Exception:
                        pass
                    for label, node in self.model_nodes.items():
                        dn = node.GetDisplayNode()
                        nid = node.GetID()
                        did = dn.GetID() if dn else None
                        if nid in ids or did in ids:
                            return label
        except Exception:
            pass
        return None

    def _vtk_ptr(self, obj):
        """Extrae el puntero C++ del repr() del objeto VTK.

        En VTK 8.x (Slicer 5.8), == entre wrappers Python compara identidad
        de objeto Python, no el puntero C++. Usando el repr se obtiene la
        dirección real y la comparación funciona en VTK 8 y 9.
        """
        if obj is None:
            return None
        try:
            return repr(obj).split('\n')[0]
        except Exception:
            return None

    def _same_vtk_obj(self, a, b):
        if a is b:
            return True
        pa, pb = self._vtk_ptr(a), self._vtk_ptr(b)
        return pa is not None and pa == pb

    def _actor_maps_to_model(self, actor, model_node):
        if not actor or not model_node:
            return False
        # 1) Comparación actor por colección del display node.
        if self._actor_belongs_to_node(actor, model_node):
            return True
        # 2) Comparación por puntero C++ del polydata del mapper.
        #    Usa _same_vtk_obj para compatibilidad con VTK 8.x (Slicer 5.8)
        #    donde == compara identidad Python, no puntero C++.
        try:
            mapper = actor.GetMapper()
            if mapper:
                node_pd = model_node.GetPolyData()
                for inp in (mapper.GetInput(), mapper.GetInputDataObject(0, 0)):
                    if self._same_vtk_obj(inp, node_pd):
                        return True
        except Exception:
            pass
        return False

    def _actor_belongs_to_node(self, picked_actor, node):
        if not picked_actor or not node:
            return False
        dn = node.GetDisplayNode()
        if not dn:
            return False
        actors = vtk.vtkActorCollection()
        try:
            dn.GetActors(actors)
        except Exception:
            return False
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            if actor == picked_actor:
                return True
            actor = actors.GetNextActor()
        return False

    def _pick_model_label_from_actor(self, actor):
        if not actor:
            return None
        # Evitar seleccionar el handle/fiduciales: solo aceptar actores de model_nodes.
        for label, node in self.model_nodes.items():
            if self._actor_maps_to_model(actor, node):
                return label
        return None

    def _label_from_dataset(self, ds):
        """Lee el label incrustado en el FieldData del polydata pickeado."""
        if ds is None:
            return None
        try:
            arr = ds.GetFieldData().GetAbstractArray("SpineLabel")
            if arr and arr.GetNumberOfTuples() > 0:
                label = arr.GetValue(0)
                if label in self.model_nodes:
                    return label
        except Exception:
            pass
        return None

    def _pick_label_with_cell_picker(self, x, y, renderer):
        """Pick robusto usando dataset/celda además de actor."""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.02)
        try:
            ok = picker.Pick(int(x), int(y), 0, renderer)
            if not ok:
                return None
            ds = None
            try:
                ds = picker.GetDataSet()
            except Exception:
                pass
            # 1) Label incrustado en FieldData — funciona en todas las versiones.
            label = self._label_from_dataset(ds)
            if label:
                return label
            # 2) Fallback: comparación por puntero C++ del polydata.
            if ds:
                for label, node in self.model_nodes.items():
                    if self._same_vtk_obj(ds, node.GetPolyData()):
                        return label
            # 3) Fallback: comparación por actor.
            actor = picker.GetActor()
            return self._pick_model_label_from_actor(actor)
        except Exception:
            return None

    def _pick_actor_at(self, x, y, renderer):
        """Picker robusto: cell picker primero, prop picker como respaldo."""
        # cell picker se maneja aparte porque puede devolver dataset.
        label = self._pick_label_with_cell_picker(x, y, renderer)
        if label:
            return label
        for picker in (vtk.vtkPropPicker(),):
            try:
                ok = picker.Pick(int(x), int(y), 0, renderer)
                if ok:
                    actor = picker.GetActor()
                    label = self._pick_model_label_from_actor(actor)
                    if label:
                        return label
            except Exception:
                pass
        return None

    # ── Click en vista 3D ────────────────────────────────────────────────────

    def _install_click_observer(self):
        interactor = slicer.app.layoutManager().threeDWidget(0)\
                         .threeDView().interactor()
        # Prioridad alta: permite seleccionar la malla antes de que la cámara consuma el click.
        try:
            self._selection_observer = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent, self._on_3d_click, 2.0)
        except TypeError:
            self._selection_observer = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent, self._on_3d_click)
        try:
            self._mouse_move_observer = interactor.AddObserver(
                vtk.vtkCommand.MouseMoveEvent, self._on_3d_mouse_move, 1.0)
        except TypeError:
            self._mouse_move_observer = interactor.AddObserver(
                vtk.vtkCommand.MouseMoveEvent, self._on_3d_mouse_move)
        try:
            self._left_release_observer = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonReleaseEvent, self._on_3d_left_release, 1.0)
        except TypeError:
            self._left_release_observer = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonReleaseEvent, self._on_3d_left_release)
        try:
            self._key_press_observer = interactor.AddObserver(
                vtk.vtkCommand.KeyPressEvent, self._on_3d_key_press, 2.0)
        except TypeError:
            self._key_press_observer = interactor.AddObserver(
                vtk.vtkCommand.KeyPressEvent, self._on_3d_key_press)
        try:
            self._key_release_observer = interactor.AddObserver(
                vtk.vtkCommand.KeyReleaseEvent, self._on_3d_key_release, 2.0)
        except TypeError:
            self._key_release_observer = interactor.AddObserver(
                vtk.vtkCommand.KeyReleaseEvent, self._on_3d_key_release)
        self._interactor_ref = interactor

    def _remove_click_observer(self):
        if self._interactor_ref:
            for obs in [self._selection_observer, self._mouse_move_observer, self._left_release_observer, self._key_press_observer, self._key_release_observer]:
                if obs:
                    try:
                        self._interactor_ref.RemoveObserver(obs)
                    except Exception:
                        pass
        self._selection_observer = None
        self._mouse_move_observer = None
        self._left_release_observer = None
        self._key_press_observer = None
        self._key_release_observer = None

    def _space_key_pressed(self, caller):
        try:
            key = caller.GetKeySym()
        except Exception:
            key = ""
        return str(key).lower() in ("space", "spacebar")

    def _on_3d_key_press(self, caller, event):
        if not self.osteotomy_mouse_mode or not self._space_key_pressed(caller):
            return
        if not self._osteotomy_drilling:
            self._begin_osteotomy_drill(caller)
        try:
            caller.AbortFlagOn()
        except Exception:
            pass

    def _on_3d_key_release(self, caller, event):
        if not self.osteotomy_mouse_mode or not self._space_key_pressed(caller):
            return
        self._end_osteotomy_drill()
        try:
            caller.AbortFlagOn()
        except Exception:
            pass

    def _on_3d_click(self, caller, event):
        try:
            if self.osteotomy_mouse_mode:
                # V41: la broca se dispara con barra espaciadora para no arrastrar la camara.
                pass
            x, y = caller.GetEventPosition()
            label = self._display_manager_pick_label(x, y)

            if label is None:
                renderer = slicer.app.layoutManager().threeDWidget(0)\
                               .threeDView().renderWindow()\
                               .GetRenderers().GetFirstRenderer()
                label = self._pick_actor_at(x, y, renderer)

            if label:
                self._combo.setCurrentText(label)
                self._update_native_transform_interaction()
                if label == self.anchor_label and not self.anchor_motion_enabled:
                    self._update_status(f"{label} seleccionada. Activá 'Permitir mover la vértebra ancla' para editarla.")
                else:
                    pivot_world = self._motion_pivot_world(label)
                    ptxt = ""
                    if pivot_world is not None:
                        ptxt = f" | pivot/disco RAS: ({pivot_world[0]:.1f}, {pivot_world[1]:.1f}, {pivot_world[2]:.1f})"
                    self._update_status(f"Seleccionada desde 3D: {label}{ptxt}")
                try:
                    caller.AbortFlagOn()
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[Selección 3D] Error: {e}")




# ══════════════════════════════════════════════════════════════════════════════
# EXTENSIÓN SLICER — clases requeridas por ScriptedLoadableModule
# ══════════════════════════════════════════════════════════════════════════════


class SpineSimulator(ScriptedLoadableModule):
    """Módulo principal de la extensión SpineSimulator."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Spine Simulator"
        self.parent.categories = ["MOBI"]
        self.parent.dependencies = []
        self.parent.contributors = ["SpineSimulator Development Team"]
        self.parent.helpText = """
Simulador de cirugía de columna vertebral para 3D Slicer.
Permite mover segmentaciones vertebrales con cinemática inversa FABRIK,
detección de colisiones y osteotomía virtual.
Cargá los modelos STL/OBJ de las vértebras y hacé click en Iniciar.
"""
        self.parent.acknowledgementText = """
Desarrollado sobre 3D Slicer. Basado en SpineSimulator V4_1.
"""


class SpineSimulatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Widget del módulo SpineSimulator.

    Gestiona el ciclo de vida del simulador: inicio, parada y limpieza
    al cerrar la escena. El panel de control del simulador se construye
    dentro del panel lateral de Slicer (self.layout) en lugar de una
    ventana flotante separada.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._sim = None        # instancia de SpineSimulatorV3

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ── Botones de control principal ──────────────────────────────────────
        controlBox = qt.QGroupBox("Control del simulador")
        controlLay = qt.QHBoxLayout(controlBox)

        self.startButton = qt.QPushButton("▶  Iniciar simulador")
        self.startButton.setStyleSheet("font-weight:bold; padding:6px; background:#2a7; color:white;")
        self.startButton.setToolTip(
            "Detecta los modelos vertebrales cargados en la escena, los convierte "
            "a VTP y abre el panel de control del simulador."
        )
        self.startButton.clicked.connect(self.onStartButton)
        controlLay.addWidget(self.startButton)

        self.stopButton = qt.QPushButton("■  Detener")
        self.stopButton.setStyleSheet("font-weight:bold; padding:6px;")
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.onStopButton)
        controlLay.addWidget(self.stopButton)

        self.layout.addWidget(controlBox)

        # ── Placeholder para el panel del simulador ───────────────────────────
        # Cuando el simulador está activo, _sim._build_panel() crea un QWidget
        # flotante (ventana propia). Para integrarlo en el panel lateral de Slicer,
        # redirigimos el panel al layout de este widget.
        self.simPanelPlaceholder = qt.QLabel(
            "Cargá modelos vertebrales (T1, T4, L3…) y hacé click en Iniciar."
        )
        self.simPanelPlaceholder.setWordWrap(True)
        self.simPanelPlaceholder.setStyleSheet("color:#888; padding:12px;")
        self.layout.addWidget(self.simPanelPlaceholder)

        # ── Área donde se inyecta el panel del simulador ──────────────────────
        self.simPanelContainer = qt.QWidget()
        self.simPanelContainerLayout = qt.QVBoxLayout(self.simPanelContainer)
        self.simPanelContainerLayout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.simPanelContainer)
        self.simPanelContainer.hide()

        self.layout.addStretch()

        # Observers de escena para limpiar al cerrar
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent,   self.onSceneEndClose)

        self.logic = SpineSimulatorLogic()

    def cleanup(self):
        self._stop_sim()
        self.removeObservers()

    def enter(self):
        """Llamado cuando el usuario abre este módulo."""
        pass

    def exit(self):
        """Llamado cuando el usuario cambia a otro módulo.
        NO detenemos el simulador — sigue corriendo en background.
        """
        pass

    # ── Handlers de botones ───────────────────────────────────────────────────

    def onStartButton(self):
        if self._sim and self._sim.ordered_labels:
            slicer.util.infoDisplay("El simulador ya está activo. Detenelo primero.")
            return
        try:
            self._sim = SpineSimulatorV3()
            # Inyectar el panel del simulador en el layout de Slicer
            # en lugar de abrirlo como ventana flotante.
            self._inject_sim_panel()
            self._sim.start()
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.simPanelPlaceholder.hide()
            self.simPanelContainer.show()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al iniciar el simulador:\n{e}")
            logging.exception("SpineSimulator start error")
            self._sim = None

    def onStopButton(self):
        self._stop_sim()

    def _stop_sim(self):
        if self._sim:
            try:
                self._sim.stop()
            except Exception as e:
                logging.warning(f"Error al detener simulador: {e}")
            self._sim = None
        # Limpiar el panel inyectado
        while self.simPanelContainerLayout.count():
            item = self.simPanelContainerLayout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.simPanelContainer.hide()
        self.simPanelPlaceholder.show()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    # ── Inyección del panel en el layout de Slicer ────────────────────────────

    def _inject_sim_panel(self):
        """
        Modifica SpineSimulatorV3._build_panel() para que el panel Qt
        se construya dentro del layout del módulo de Slicer en lugar de
        abrirse como ventana flotante independiente.

        Estrategia: monkey-patch de _build_panel para que cree el contenido
        en simPanelContainerLayout en vez de en un QWidget top-level.
        """
        container_layout = self.simPanelContainerLayout
        sim = self._sim

        def _build_panel_embedded():
            """Versión embebida de _build_panel: usa el layout del módulo Slicer."""

            # Scroll area para que el contenido no quede cortado
            scroll = qt.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
            container_layout.addWidget(scroll)

            content = qt.QWidget()
            scroll.setWidget(content)
            root = qt.QVBoxLayout(content)
            root.setSpacing(8)
            root.setContentsMargins(8, 8, 8, 8)

            # ── Selector de vértebra activa ──
            selBox = qt.QGroupBox("Vértebra activa")
            selLay = qt.QHBoxLayout(selBox)
            sim._combo = qt.QComboBox()
            for l in sim.ordered_labels:
                sim._combo.addItem(l)
            sim._combo.currentTextChanged.connect(sim._on_vertebra_selected)
            selLay.addWidget(sim._combo)
            sim._anchor_label_widget = qt.QLabel(f"Ancla: {sim.anchor_label}")
            sim._anchor_label_widget.setStyleSheet("color:#777;font-size:11px")
            selLay.addWidget(sim._anchor_label_widget)
            root.addWidget(selBox)

            # ── Ancla ──
            anchorBox = qt.QGroupBox("Ancla / raíz cinemática")
            anchorLay = qt.QFormLayout(anchorBox)
            sim._anchor_combo = qt.QComboBox()
            for l in sim.ordered_labels:
                sim._anchor_combo.addItem(l)
            if sim.anchor_label in sim.ordered_labels:
                sim._anchor_combo.setCurrentText(sim.anchor_label)
            sim._anchor_combo.currentTextChanged.connect(sim._on_anchor_selected)
            anchorLay.addRow("Vértebra ancla", sim._anchor_combo)
            sim._anchor_motion_check = qt.QCheckBox("Permitir mover la vértebra ancla")
            sim._anchor_motion_check.setChecked(False)
            sim._anchor_motion_check.toggled.connect(sim._on_anchor_motion_toggled)
            anchorLay.addRow(sim._anchor_motion_check)
            root.addWidget(anchorBox)

            # ── Rotación ──
            rotBox = qt.QGroupBox("Rotación (control principal)")
            rotLay = qt.QFormLayout(rotBox)
            sim._rot_widgets = {}
            rot_axes = [
                ("Flexión / extensión  (X)", "rx", (-30, 30)),
                ("Rotación axial          (Y)", "ry", (-45, 45)),
                ("Inclinación lateral  (Z)", "rz", (-45, 45)),
            ]
            for name, key, (mn, mx) in rot_axes:
                row = qt.QHBoxLayout()
                sl = qt.QSlider(qt.Qt.Horizontal)
                sl.setRange(int(mn*10), int(mx*10))
                sl.setValue(0)
                sb = qt.QDoubleSpinBox()
                sb.setRange(mn, mx)
                sb.setSingleStep(0.5)
                sb.setDecimals(1)
                sb.setSuffix("°")
                sb.setFixedWidth(72)
                sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/10.0))
                sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*10)))
                sl.valueChanged.connect(partial(sim._on_rot_slider, key))
                row.addWidget(sl)
                row.addWidget(sb)
                rotLay.addRow(name, row)
                sim._rot_widgets[key] = (sl, sb)
            root.addWidget(rotBox)

            # ── Handle nativo ──
            nativeBox = qt.QGroupBox("Interacción 3D nativa")
            nativeLay = qt.QFormLayout(nativeBox)
            sim._native_interaction_check = qt.QCheckBox("Activar handle nativo")
            sim._native_interaction_check.setChecked(sim.native_transform_interaction_enabled)
            sim._native_interaction_check.toggled.connect(sim._on_native_interaction_toggled)
            nativeLay.addRow(sim._native_interaction_check)
            sim._native_rotation_only_check = qt.QCheckBox("Solo rotación en el handle 3D")
            sim._native_rotation_only_check.setChecked(sim.native_rotation_only)
            sim._native_rotation_only_check.toggled.connect(sim._on_native_rotation_only_toggled)
            nativeLay.addRow(sim._native_rotation_only_check)
            sim._native_handle_scale_spin = qt.QDoubleSpinBox()
            sim._native_handle_scale_spin.setRange(0.2, 5.0)
            sim._native_handle_scale_spin.setSingleStep(0.1)
            sim._native_handle_scale_spin.setDecimals(1)
            sim._native_handle_scale_spin.setValue(sim.native_handle_scale)
            sim._native_handle_scale_spin.valueChanged.connect(sim._on_native_handle_scale_changed)
            nativeLay.addRow("Tamaño handle", sim._native_handle_scale_spin)
            sim._native_handle_offset_check = qt.QCheckBox("Offset visual a la derecha")
            sim._native_handle_offset_check.setChecked(sim.native_handle_screen_offset_enabled)
            sim._native_handle_offset_check.toggled.connect(sim._on_native_handle_offset_toggled)
            nativeLay.addRow(sim._native_handle_offset_check)
            sim._native_handle_offset_spin = qt.QDoubleSpinBox()
            sim._native_handle_offset_spin.setRange(0.0, 250.0)
            sim._native_handle_offset_spin.setSingleStep(5.0)
            sim._native_handle_offset_spin.setDecimals(1)
            sim._native_handle_offset_spin.setSuffix(" mm")
            sim._native_handle_offset_spin.setValue(sim.native_handle_screen_offset_mm)
            sim._native_handle_offset_spin.valueChanged.connect(sim._on_native_handle_offset_changed)
            nativeLay.addRow("Offset derecha", sim._native_handle_offset_spin)
            root.addWidget(nativeBox)

            # ── Traslación ──
            transBox = qt.QGroupBox("Traslación — ajuste fino (mm)")
            transBox.setCheckable(True)
            transBox.setChecked(False)
            transLay = qt.QFormLayout(transBox)
            sim._trans_widgets = {}
            trans_axes = [
                ("Lateral  (X)", "tx", (-40, 40)),
                ("Ant/Post (Y)", "ty", (-40, 40)),
                ("Craneal  (Z)", "tz", (-50, 50)),
            ]
            for name, key, (mn, mx) in trans_axes:
                row = qt.QHBoxLayout()
                sl = qt.QSlider(qt.Qt.Horizontal)
                sl.setRange(int(mn*10), int(mx*10))
                sl.setValue(0)
                sb = qt.QDoubleSpinBox()
                sb.setRange(mn, mx)
                sb.setSingleStep(0.5)
                sb.setDecimals(1)
                sb.setSuffix(" mm")
                sb.setFixedWidth(72)
                sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/10.0))
                sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*10)))
                sl.valueChanged.connect(partial(sim._on_trans_slider, key))
                row.addWidget(sl)
                row.addWidget(sb)
                transLay.addRow(name, row)
                sim._trans_widgets[key] = (sl, sb)
            root.addWidget(transBox)

            # ── Dinámica ──
            dynBox = qt.QGroupBox("Dinámica distribuida")
            dynLay = qt.QFormLayout(dynBox)
            sim._dynamic_check = qt.QCheckBox("Mover vecinas al mover una vértebra")
            sim._dynamic_check.setChecked(sim.dynamic_enabled)
            sim._dynamic_check.toggled.connect(sim._on_dynamic_enabled_changed)
            dynLay.addRow(sim._dynamic_check)
            sim._radius_spin = qt.QSpinBox()
            sim._radius_spin.setRange(0, 6)
            sim._radius_spin.setValue(int(sim.influence_radius))
            sim._radius_spin.setSuffix(" niveles")
            sim._radius_spin.valueChanged.connect(sim._on_influence_radius_changed)
            dynLay.addRow("Alcance", sim._radius_spin)
            sim._decay_spin = qt.QDoubleSpinBox()
            sim._decay_spin.setRange(0.0, 1.0)
            sim._decay_spin.setSingleStep(0.05)
            sim._decay_spin.setDecimals(2)
            sim._decay_spin.setValue(float(sim.influence_decay))
            sim._decay_spin.valueChanged.connect(sim._on_influence_decay_changed)
            dynLay.addRow("Suavidad vecina", sim._decay_spin)
            sim._chain_check = qt.QCheckBox("Cadena tipo huesos")
            sim._chain_check.setChecked(sim.kinematic_chain_enabled)
            sim._chain_check.toggled.connect(sim._on_kinematic_chain_changed)
            dynLay.addRow(sim._chain_check)
            sim._local_bend_spin = qt.QDoubleSpinBox()
            sim._local_bend_spin.setRange(0.0, 0.6)
            sim._local_bend_spin.setSingleStep(0.05)
            sim._local_bend_spin.setDecimals(2)
            sim._local_bend_spin.setValue(float(sim.local_bend_fraction))
            sim._local_bend_spin.valueChanged.connect(sim._on_local_bend_changed)
            dynLay.addRow("Curvatura local", sim._local_bend_spin)
            root.addWidget(dynBox)

            # ── Colisiones ──
            colBox = qt.QGroupBox("Colisiones VTP")
            colLay = qt.QFormLayout(colBox)
            sim._collision_check = qt.QCheckBox("Detectar contacto entre mallas VTP")
            sim._collision_check.setChecked(sim.collision_enabled)
            sim._collision_check.toggled.connect(sim._on_collision_enabled_changed)
            colLay.addRow(sim._collision_check)
            sim._collision_blocking_check = qt.QCheckBox("Bloquear movimiento al colisionar")
            sim._collision_blocking_check.setChecked(sim.collision_blocking_enabled)
            sim._collision_blocking_check.toggled.connect(sim._on_collision_blocking_changed)
            colLay.addRow(sim._collision_blocking_check)
            sim._collision_margin_spin = qt.QDoubleSpinBox()
            sim._collision_margin_spin.setRange(0.0, 5.0)
            sim._collision_margin_spin.setSingleStep(0.1)
            sim._collision_margin_spin.setDecimals(2)
            sim._collision_margin_spin.setSuffix(" mm")
            sim._collision_margin_spin.setValue(float(sim.collision_margin_mm))
            sim._collision_margin_spin.valueChanged.connect(sim._on_collision_margin_changed)
            colLay.addRow("Margen contacto", sim._collision_margin_spin)
            sim._collision_heatmap_check = qt.QCheckBox("Mostrar contacto visual")
            sim._collision_heatmap_check.setChecked(sim.collision_heatmap_enabled)
            sim._collision_heatmap_check.toggled.connect(sim._on_collision_heatmap_changed)
            colLay.addRow(sim._collision_heatmap_check)
            sim._collision_heatmap_mode_combo = qt.QComboBox()
            sim._collision_heatmap_mode_combo.addItem("Superficie roja", "PATCH")
            sim._collision_heatmap_mode_combo.addItem("Bolitas rojas", "SPHERES")
            sim._collision_heatmap_mode_combo.addItem("Mapa de calor suave", "SURFACE")
            mode_index = 2 if sim.collision_heatmap_mode == "SURFACE" else (1 if sim.collision_heatmap_mode == "SPHERES" else 0)
            sim._collision_heatmap_mode_combo.setCurrentIndex(mode_index)
            sim._collision_heatmap_mode_combo.currentIndexChanged.connect(sim._on_collision_heatmap_mode_changed)
            colLay.addRow("Visualización", sim._collision_heatmap_mode_combo)
            recalibBtn = qt.QPushButton("Recalibrar postura actual")
            recalibBtn.clicked.connect(sim._on_collision_recalibrate_clicked)
            colLay.addRow(recalibBtn)
            clearBtn = qt.QPushButton("Limpiar marcas de contacto")
            clearBtn.clicked.connect(sim._on_clear_contact_marks_clicked)
            colLay.addRow(clearBtn)
            root.addWidget(colBox)

            # ── Osteotomía ──
            ostBox = qt.QGroupBox("Osteotomía virtual VTP")
            ostLay = qt.QFormLayout(ostBox)
            sim._osteotomy_mouse_check = qt.QCheckBox("Modo broca con mouse")
            sim._osteotomy_mouse_check.setChecked(sim.osteotomy_mouse_mode)
            sim._osteotomy_mouse_check.toggled.connect(sim._on_osteotomy_mouse_mode_changed)
            ostLay.addRow(sim._osteotomy_mouse_check)
            sim._osteotomy_continuous_check = qt.QCheckBox("Drill continuo al arrastrar")
            sim._osteotomy_continuous_check.setChecked(sim.osteotomy_continuous_drill)
            sim._osteotomy_continuous_check.toggled.connect(sim._on_osteotomy_continuous_changed)
            ostLay.addRow(sim._osteotomy_continuous_check)
            sim._osteotomy_radius_spin = qt.QDoubleSpinBox()
            sim._osteotomy_radius_spin.setRange(0.5, 20.0)
            sim._osteotomy_radius_spin.setSingleStep(0.5)
            sim._osteotomy_radius_spin.setDecimals(1)
            sim._osteotomy_radius_spin.setSuffix(" mm")
            sim._osteotomy_radius_spin.setValue(float(sim.osteotomy_radius_mm))
            sim._osteotomy_radius_spin.valueChanged.connect(sim._on_osteotomy_radius_changed)
            ostLay.addRow("Radio broca", sim._osteotomy_radius_spin)
            placeDrillBtn = qt.QPushButton("Broca en contacto activo")
            placeDrillBtn.clicked.connect(sim._on_osteotomy_place_from_contact)
            ostLay.addRow(placeDrillBtn)
            applyDrillBtn = qt.QPushButton("Aplicar osteotomía")
            applyDrillBtn.clicked.connect(sim._on_osteotomy_apply)
            ostLay.addRow(applyDrillBtn)
            resetDrillBtn = qt.QPushButton("Revertir osteotomía activa")
            resetDrillBtn.clicked.connect(sim._on_osteotomy_reset_active)
            ostLay.addRow(resetDrillBtn)
            root.addWidget(ostBox)

            # ── Pivotes ──
            pivBox = qt.QGroupBox("Pivotes anatómicos")
            pivLay = qt.QFormLayout(pivBox)
            sim._pivot_mode_combo = qt.QComboBox()
            sim._pivot_mode_combo.addItem("Cuerpo vertebral por densidad (+Y anterior)", "BODY_DENSITY_POS_Y")
            sim._pivot_mode_combo.addItem("Cuerpo vertebral por densidad (-Y anterior)", "BODY_DENSITY_NEG_Y")
            sim._pivot_mode_combo.addItem("Centro bound completo", "BOUNDS_CENTER")
            sim._pivot_mode_combo.addItem("Centro de masa de toda la malla", "CENTER_OF_MASS")
            sim._pivot_mode_combo.addItem("Manual: usar fiduciales editables", "MANUAL_FIDUCIALS")
            sim._pivot_mode_combo.currentIndexChanged.connect(sim._on_pivot_mode_changed)
            pivLay.addRow("Modo pivot", sim._pivot_mode_combo)
            sim._pivot_check = qt.QCheckBox("Mostrar centros de cuerpos Pivot_XX")
            sim._pivot_check.setChecked(sim.show_pivot_fiducials)
            sim._pivot_check.toggled.connect(sim._on_pivot_visibility_changed)
            pivLay.addRow(sim._pivot_check)
            sim._disc_check = qt.QCheckBox("Mostrar discos Disc_XX_YY celestes")
            sim._disc_check.setChecked(sim.show_disc_fiducials)
            sim._disc_check.toggled.connect(sim._on_disc_visibility_changed)
            pivLay.addRow(sim._disc_check)
            sim._use_disc_check = qt.QCheckBox("Usar discos como pivots de movimiento")
            sim._use_disc_check.setChecked(sim.use_disc_pivots)
            sim._use_disc_check.toggled.connect(sim._on_use_disc_pivots_changed)
            pivLay.addRow(sim._use_disc_check)
            recalcDiscBtn = qt.QPushButton("Recalcular discos entre cuerpos")
            recalcDiscBtn.clicked.connect(sim._on_recalculate_discs_clicked)
            pivLay.addRow(recalcDiscBtn)
            useDiscBtn = qt.QPushButton("Activar discos actuales como pivots")
            useDiscBtn.clicked.connect(sim._on_use_current_discs_clicked)
            pivLay.addRow(useDiscBtn)
            recalcPivotBtn = qt.QPushButton("Recalcular centros automáticos")
            recalcPivotBtn.clicked.connect(sim._on_recalculate_pivots_clicked)
            pivLay.addRow(recalcPivotBtn)
            useManualBtn = qt.QPushButton("Aplicar Pivot_XX manuales")
            useManualBtn.clicked.connect(sim._on_use_fiducials_as_pivots_clicked)
            pivLay.addRow(useManualBtn)
            root.addWidget(pivBox)

            # ── Botones finales ──
            btnRow = qt.QHBoxLayout()
            for txt, fn in [("Reset vértebra", sim._on_reset_active),
                             ("Reset todo",     sim.reset_all),
                             ("Exportar",       sim._on_export)]:
                b = qt.QPushButton(txt)
                b.clicked.connect(fn)
                btnRow.addWidget(b)
            diagBtn = qt.QPushButton("Diagnóstico")
            diagBtn.clicked.connect(sim._on_self_test_clicked)
            btnRow.addWidget(diagBtn)
            root.addLayout(btnRow)

            sim._status_lbl = qt.QLabel("Click en un modelo 3D para seleccionar.")
            sim._status_lbl.setStyleSheet("color:#777;font-size:11px")
            root.addWidget(sim._status_lbl)

            # En la extensión NO se llama self._panel.show() porque el panel
            # vive dentro del layout del módulo de Slicer, no es una ventana flotante.
            sim._panel = scroll  # referencia para compatibilidad con sim.stop()

        # Monkey-patch: reemplazar _build_panel por la versión embebida
        import types
        sim._build_panel = types.MethodType(lambda self: _build_panel_embedded(), sim)

    # ── Observers de escena ───────────────────────────────────────────────────

    def onSceneStartClose(self, caller, event):
        """Al cerrar la escena, detener el simulador limpiamente."""
        self._stop_sim()

    def onSceneEndClose(self, caller, event):
        """Después de cerrar la escena, resetear la UI."""
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)


class SpineSimulatorLogic(ScriptedLoadableModuleLogic):
    """
    Lógica del módulo. En este caso es un wrapper mínimo —
    toda la lógica real vive en SpineSimulatorV3.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class SpineSimulatorTest(ScriptedLoadableModuleTest):
    """Tests básicos del módulo."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_moduleLoads()

    def test_moduleLoads(self):
        self.delayDisplay("Verificando que el módulo carga correctamente...")
        self.assertIsNotNone(slicer.modules.spinesimulator)
        self.delayDisplay("OK — módulo cargado.")
