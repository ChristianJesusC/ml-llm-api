import json
import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from ml_model import VocationalMLPredictor

class QuestionnaireSession:
    """Clase para manejar sesiones individuales de cuestionario"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.respuestas = {}
        self.fase_actual = 'fase1'
        self.rama_detectada = None
        self.categoria_detectada = None
        self.capacidad_estimada = None
        
        self.timestamp_inicio = datetime.now()
        self.timestamp_fin = None
        
        self.estadisticas = {
            'preguntas_respondidas': 0,
            'progreso_fase1': 0,
            'progreso_fase2': 0,
            'puede_finalizar': False,
            'confianza_prediccion': 0.0,
            'tiempo_estimado_restante': 0
        }
        
        self.historial_preguntas = []
        
    def agregar_respuesta(self, pregunta_id: str, respuesta: str, metadata: Dict = None):
        """Agrega una respuesta del estudiante"""
        self.respuestas[pregunta_id] = respuesta
        
        self.historial_preguntas.append({
            'pregunta_id': pregunta_id,
            'respuesta': respuesta,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        
        self.estadisticas['preguntas_respondidas'] = len(self.respuestas)
        
        # Actualizar progreso por fase
        if pregunta_id.startswith('fase1_') or any(pregunta_id.endswith(f'{i:03d}') for i in range(1, 81)):
            self.estadisticas['progreso_fase1'] = len([p for p in self.respuestas if 'fase1' in p or any(p.endswith(f'{i:03d}') for i in range(1, 81))])
        else:
            self.estadisticas['progreso_fase2'] = len([p for p in self.respuestas if 'fase2' in p])
    
    def to_dict(self):
        """Serializa la sesión a diccionario"""
        return {
            'session_id': self.session_id,
            'respuestas': self.respuestas,
            'fase_actual': self.fase_actual,
            'rama_detectada': self.rama_detectada,
            'categoria_detectada': self.categoria_detectada,
            'capacidad_estimada': self.capacidad_estimada,
            'estadisticas': self.estadisticas,
            'timestamp_inicio': self.timestamp_inicio.isoformat(),
            'timestamp_fin': self.timestamp_fin.isoformat() if self.timestamp_fin else None,
            'total_preguntas_respondidas': len(self.respuestas)
        }


class AdaptiveQuestionnaire:
    """Sistema de Cuestionario Adaptativo principal"""
    
    def __init__(self, 
                 ml_predictor: VocationalMLPredictor,
                 preguntas_fase1_path: str = 'preguntas-fase1.json',
                 preguntas_fase2_paths: Dict[str, str] = None):
        
        self.ml_predictor = ml_predictor
        self.sesiones = {}
        
        # Configuración adaptativa
        self.config = {
            'min_preguntas_fase1': 30,     # Mínimo de Fase 1 antes de evaluar
            'max_preguntas_fase1': 80,     # Máximo de Fase 1
            'min_preguntas_fase2': 20,     # Mínimo de Fase 2 
            'max_preguntas_fase2': 40,     # Máximo de Fase 2
            'min_total_preguntas': 50,     # Mínimo total para finalizar
            'max_total_preguntas': 120,    # Máximo total absoluto
            'umbral_confianza': 0.75,      # Umbral para parar temprano
            'evaluacion_cada_n': 5        # Evaluar progreso cada N preguntas
        }
        
        # Cargar bancos de preguntas
        self._cargar_preguntas(preguntas_fase1_path, preguntas_fase2_paths or {})
        
        print("✅ AdaptiveQuestionnaire inicializado")
        print(f"   📊 Preguntas Fase 1: {len(self.preguntas_fase1)}")
        print(f"   📊 Preguntas Fase 2: {len(self.preguntas_fase2)}")
    
    def _cargar_preguntas(self, fase1_path: str, fase2_paths: Dict[str, str]):
        """Carga los bancos de preguntas"""
        
        # Cargar Fase 1
        try:
            with open(fase1_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.preguntas_fase1 = {p['id']: p for p in data['preguntas']}
        except Exception as e:
            print(f"⚠️  Error cargando Fase 1: {e}")
            self.preguntas_fase1 = {}
        
        # Cargar Fase 2 (múltiples archivos por rama)
        self.preguntas_fase2 = {}
        
        # Paths por defecto si no se proporcionan
        if not fase2_paths:
            fase2_paths = {
                'Salud': 'rama1_salud_preguntas_fase2.json',
                'Ingenieria': 'rama2_ingenieria_preguntas_fase2.json', 
                'Negocios': 'rama3_negocios_preguntas_fase2.json',
                'Oficios_Tecnicos': 'rama4_oficios_tecnicos_preguntas_fase2.json',
                'Oficios_Basicos': 'rama5_oficios_basicos_preguntas_fase2.json'
            }
        
        for rama, path in fase2_paths.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.preguntas_fase2[rama] = {p['id']: p for p in data['preguntas']}
                print(f"   ✅ {rama}: {len(self.preguntas_fase2[rama])} preguntas")
            except Exception as e:
                print(f"   ⚠️  Error cargando {rama}: {e}")
                self.preguntas_fase2[rama] = {}
    
    def iniciar_sesion(self) -> QuestionnaireSession:
        """Inicia una nueva sesión de cuestionario"""
        sesion = QuestionnaireSession()
        self.sesiones[sesion.session_id] = sesion
        
        print(f"📋 Nueva sesión iniciada: {sesion.session_id}")
        return sesion
    
    def obtener_siguiente_pregunta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la siguiente pregunta de forma adaptativa
        
        Returns:
            Dict con pregunta y metadata o None si debe terminar
        """
        if session_id not in self.sesiones:
            raise ValueError(f"Sesión {session_id} no encontrada")
        
        sesion = self.sesiones[session_id]
        n_respuestas = len(sesion.respuestas)
        
        print(f"📝 Obteniendo siguiente pregunta (total: {n_respuestas})")
        
        # Evaluar si debe continuar o terminar
        debe_continuar, razon = self._evaluar_continuacion(sesion)
        
        if not debe_continuar:
            print(f"✅ Sistema decidió terminar: {razon}")
            return None
        
        # Seleccionar siguiente pregunta
        pregunta_id = self._seleccionar_pregunta(sesion)
        
        if not pregunta_id:
            print("⚠️  No se pudo seleccionar siguiente pregunta")
            return None
        
        # Obtener pregunta del banco correspondiente
        pregunta = self._obtener_pregunta_por_id(pregunta_id, sesion)
        
        if not pregunta:
            print(f"⚠️  Pregunta {pregunta_id} no encontrada")
            return None
        
        # Preparar metadata de respuesta
        metadata = {
            'pregunta_id': pregunta_id,
            'fase': sesion.fase_actual,
            'progreso': self._calcular_progreso(sesion),
            'puede_finalizar': n_respuestas >= self.config['min_total_preguntas'],
            'tiempo_estimado': self._estimar_tiempo_restante(sesion)
        }
        
        return {
            'pregunta': pregunta,
            'metadata': metadata
        }
    
    def enviar_respuesta(self, session_id: str, pregunta_id: str, respuesta: str) -> Dict[str, Any]:
        """
        Registra una respuesta del estudiante
        
        Returns:
            Dict con resultado y metadata actualizada
        """
        if session_id not in self.sesiones:
            raise ValueError(f"Sesión {session_id} no encontrada")
        
        if respuesta not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError("Respuesta debe ser A, B, C, D o E")
        
        sesion = self.sesiones[session_id]
        
        # Registrar respuesta
        sesion.agregar_respuesta(pregunta_id, respuesta)
        
        n_respuestas = len(sesion.respuestas)
        print(f"📝 Respuesta registrada: {pregunta_id} → {respuesta} (total: {n_respuestas})")
        
        # Evaluar progreso cada N preguntas
        if n_respuestas % self.config['evaluacion_cada_n'] == 0:
            self._evaluar_progreso_sesion(sesion)
        
        # Generar predicción parcial si es apropiado
        prediccion_parcial = None
        if n_respuestas >= self.config['min_total_preguntas']:
            try:
                resultado = self.ml_predictor.generar_recomendaciones_completas(
                    sesion.respuestas, top_n=3
                )
                prediccion_parcial = {
                    'top_3': [(r['nombre'], r['match_score_porcentaje']) 
                             for r in resultado['recomendaciones']],
                    'capacidad': resultado['capacidad_academica']['score'],
                    'categoria': resultado['capacidad_academica']['categoria']
                }
            except Exception as e:
                print(f"⚠️  Error generando predicción parcial: {e}")
        
        # Evaluar si puede/debe continuar
        debe_continuar, razon = self._evaluar_continuacion(sesion)
        
        return {
            'respuesta_registrada': True,
            'total_respuestas': n_respuestas,
            'fase_actual': sesion.fase_actual,
            'progreso': self._calcular_progreso(sesion),
            'debe_continuar': debe_continuar,
            'puede_finalizar': n_respuestas >= self.config['min_total_preguntas'],
            'razon_estado': razon,
            'prediccion_parcial': prediccion_parcial,
            'estadisticas': sesion.estadisticas
        }
    
    def _evaluar_continuacion(self, sesion: QuestionnaireSession) -> Tuple[bool, str]:
        """Evalúa si el cuestionario debe continuar o terminar"""
        
        n_respuestas = len(sesion.respuestas)
        
        # Caso 1: Muy pocas respuestas
        if n_respuestas < self.config['min_total_preguntas']:
            return True, f"Mínimo {self.config['min_total_preguntas']} preguntas requeridas"
        
        # Caso 2: Máximo absoluto alcanzado
        if n_respuestas >= self.config['max_total_preguntas']:
            return False, f"Máximo de {self.config['max_total_preguntas']} preguntas alcanzado"
        
        # Caso 3: Evaluar confianza de predicción (solo si tenemos suficientes respuestas)
        if n_respuestas >= 60:  # Evaluar confianza solo con suficientes respuestas
            try:
                resultado = self.ml_predictor.generar_recomendaciones_completas(
                    sesion.respuestas, top_n=3
                )
                
                # Calcular diferencia entre top 2
                if len(resultado['recomendaciones']) >= 2:
                    score_1 = resultado['recomendaciones'][0]['match_score']
                    score_2 = resultado['recomendaciones'][1]['match_score']
                    diferencia = score_1 - score_2
                    
                    sesion.estadisticas['confianza_prediccion'] = score_1
                    
                    # Si hay alta confianza, podemos parar
                    if score_1 > self.config['umbral_confianza'] and diferencia > 0.15:
                        return False, f"Alta confianza alcanzada (top: {score_1:.1%}, diff: {diferencia:.1%})"
                
            except Exception as e:
                print(f"⚠️  Error evaluando confianza: {e}")
        
        # Caso 4: Evaluar fase específica
        if sesion.fase_actual == 'fase1':
            fase1_respondidas = sesion.estadisticas['progreso_fase1']
            
            # Cambiar a Fase 2 si se completó suficiente Fase 1
            if fase1_respondidas >= self.config['min_preguntas_fase1']:
                return True, "Listo para Fase 2"
            elif fase1_respondidas >= self.config['max_preguntas_fase1']:
                return True, "Máximo Fase 1 alcanzado"
            else:
                return True, f"Continuando Fase 1 ({fase1_respondidas}/{self.config['min_preguntas_fase1']})"
        
        elif sesion.fase_actual == 'fase2':
            fase2_respondidas = sesion.estadisticas['progreso_fase2']
            
            if fase2_respondidas >= self.config['max_preguntas_fase2']:
                return False, "Máximo Fase 2 alcanzado"
            elif fase2_respondidas >= self.config['min_preguntas_fase2'] and n_respuestas >= 80:
                return False, "Fase 2 completa"
            else:
                return True, f"Continuando Fase 2 ({fase2_respondidas}/{self.config['min_preguntas_fase2']})"
        
        return True, "Continuando cuestionario"
    
    def _seleccionar_pregunta(self, sesion: QuestionnaireSession) -> Optional[str]:
        """Selecciona la siguiente pregunta de forma adaptativa"""
        
        preguntas_respondidas = set(sesion.respuestas.keys())
        
        # Determinar fase actual basada en progreso
        if sesion.estadisticas['progreso_fase1'] < self.config['min_preguntas_fase1']:
            sesion.fase_actual = 'fase1'
        elif not sesion.categoria_detectada:
            # Evaluar capacidad para determinar categoría
            self._evaluar_progreso_sesion(sesion)
            if sesion.categoria_detectada:
                sesion.fase_actual = 'fase2'
            else:
                sesion.fase_actual = 'fase1'  # Continuar Fase 1
        else:
            sesion.fase_actual = 'fase2'
        
        # Selección por fase
        if sesion.fase_actual == 'fase1':
            return self._seleccionar_pregunta_fase1(preguntas_respondidas)
        else:
            return self._seleccionar_pregunta_fase2(sesion, preguntas_respondidas)
    
    def _seleccionar_pregunta_fase1(self, preguntas_respondidas: set) -> Optional[str]:
        """Selecciona pregunta de Fase 1 (secuencial con algunos saltos)"""
        
        # Obtener preguntas disponibles de Fase 1
        disponibles = []
        for i in range(1, 81):  # UNI-001 a UNI-080
            pregunta_id = f'UNI-{i:03d}'
            if pregunta_id not in preguntas_respondidas and pregunta_id in self.preguntas_fase1:
                disponibles.append((i, pregunta_id))
        
        if not disponibles:
            return None
        
        # Estrategia: Mayormente secuencial pero con some randomización
        if len(preguntas_respondidas) < 20:
            # Primeras 20: estrictamente secuencial
            disponibles.sort()
            return disponibles[0][1]
        else:
            # Después: mayor randomización, pero priorizando las primeras
            disponibles.sort()
            # 70% probabilidad de tomar las primeras 5, 30% cualquier otra
            if random.random() < 0.7 and len(disponibles) >= 5:
                return random.choice(disponibles[:5])[1]
            else:
                return random.choice(disponibles)[1]
    
    def _seleccionar_pregunta_fase2(self, sesion: QuestionnaireSession, preguntas_respondidas: set) -> Optional[str]:
        """Selecciona pregunta de Fase 2 basada en categoría/rama detectada"""
        
        # Determinar qué banco usar
        if sesion.categoria_detectada == 'carreras' and sesion.rama_detectada:
            banco_key = sesion.rama_detectada
        elif sesion.categoria_detectada == 'oficios_tecnicos':
            banco_key = 'Oficios_Tecnicos'
        elif sesion.categoria_detectada == 'oficios_basicos':
            banco_key = 'Oficios_Basicos'
        else:
            # Fallback: elegir banco aleatoriamente
            banco_key = random.choice(['Salud', 'Ingenieria', 'Negocios'])
        
        if banco_key not in self.preguntas_fase2:
            print(f"⚠️  Banco {banco_key} no disponible")
            return None
        
        banco = self.preguntas_fase2[banco_key]
        
        # Seleccionar pregunta disponible
        disponibles = [pid for pid in banco.keys() if pid not in preguntas_respondidas]
        
        if not disponibles:
            return None
        
        # Por ahora, selección aleatoria
        # TODO: Implementar selección inteligente basada en importancia
        return random.choice(disponibles)
    
    def _evaluar_progreso_sesion(self, sesion: QuestionnaireSession):
        """Evalúa el progreso y actualiza metadatos de la sesión"""
        
        n_respuestas = len(sesion.respuestas)
        
        # Solo evaluar si tenemos suficientes respuestas de Fase 1
        if sesion.estadisticas['progreso_fase1'] >= 20:
            try:
                # Separar respuestas de Fase 1
                respuestas_fase1 = {k: v for k, v in sesion.respuestas.items() 
                                   if 'fase1' in k or any(k.endswith(f'{i:03d}') for i in range(1, 81))}
                
                # Evaluar capacidad
                if len(respuestas_fase1) >= 20:
                    capacidad = self.ml_predictor.predecir_capacidad_academica(respuestas_fase1)
                    categoria = self.ml_predictor.determinar_categoria_estudiante(capacidad)
                    
                    sesion.capacidad_estimada = capacidad
                    sesion.categoria_detectada = categoria
                    
                    print(f"   📊 Capacidad estimada: {capacidad:.3f} → {categoria}")
                    
                    # Si es universitario, detectar rama
                    if categoria == 'carreras' and len(respuestas_fase1) >= 30:
                        rama = self.ml_predictor.predecir_rama_universitaria(respuestas_fase1)
                        sesion.rama_detectada = rama
                        print(f"   🌳 Rama detectada: {rama}")
                
            except Exception as e:
                print(f"⚠️  Error evaluando progreso: {e}")
    
    def _obtener_pregunta_por_id(self, pregunta_id: str, sesion: QuestionnaireSession) -> Optional[Dict]:
        """Obtiene una pregunta específica del banco correspondiente"""
        
        # Intentar Fase 1 primero
        if pregunta_id in self.preguntas_fase1:
            return self.preguntas_fase1[pregunta_id]
        
        # Buscar en Fase 2
        for banco in self.preguntas_fase2.values():
            if pregunta_id in banco:
                return banco[pregunta_id]
        
        return None
    
    def _calcular_progreso(self, sesion: QuestionnaireSession) -> Dict[str, Any]:
        """Calcula el progreso actual del cuestionario"""
        
        n_total = len(sesion.respuestas)
        n_fase1 = sesion.estadisticas['progreso_fase1']
        n_fase2 = sesion.estadisticas['progreso_fase2']
        
        # Progreso estimado (no lineal)
        if n_total < self.config['min_total_preguntas']:
            porcentaje = (n_total / self.config['min_total_preguntas']) * 70
        else:
            # Entre mínimo y máximo
            progreso_adicional = (n_total - self.config['min_total_preguntas']) / \
                                (self.config['max_total_preguntas'] - self.config['min_total_preguntas'])
            porcentaje = 70 + (progreso_adicional * 30)
        
        return {
            'total_preguntas': n_total,
            'fase1_completadas': n_fase1,
            'fase2_completadas': n_fase2,
            'porcentaje_estimado': min(round(porcentaje, 1), 95),  # Nunca 100% hasta terminar
            'fase_actual': sesion.fase_actual,
            'min_total_requerido': self.config['min_total_preguntas'],
            'max_total_estimado': self.config['max_total_preguntas']
        }
    
    def _estimar_tiempo_restante(self, sesion: QuestionnaireSession) -> int:
        """Estima tiempo restante en minutos"""
        
        n_respuestas = len(sesion.respuestas)
        tiempo_promedio_por_pregunta = 1.5  # minutos
        
        # Estimar preguntas restantes
        if n_respuestas < self.config['min_total_preguntas']:
            preguntas_restantes = self.config['min_total_preguntas'] - n_respuestas
        else:
            # Basado en confianza de predicción
            preguntas_restantes = max(5, min(20, 
                int((self.config['max_total_preguntas'] - n_respuestas) * 0.5)))
        
        return int(preguntas_restantes * tiempo_promedio_por_pregunta)
    
    def finalizar_cuestionario(self, session_id: str) -> Dict[str, Any]:
        """
        Finaliza el cuestionario y genera recomendaciones completas
        
        Returns:
            Dict con recomendaciones finales
        """
        if session_id not in self.sesiones:
            raise ValueError(f"Sesión {session_id} no encontrada")
        
        sesion = self.sesiones[session_id]
        
        # Validar que tiene suficientes respuestas
        if len(sesion.respuestas) < self.config['min_total_preguntas']:
            raise ValueError(f"Se necesitan al menos {self.config['min_total_preguntas']} respuestas")
        
        # Marcar como finalizada
        sesion.timestamp_fin = datetime.now()
        
        print(f"🎓 Finalizando cuestionario: {session_id}")
        print(f"   📊 Total respuestas: {len(sesion.respuestas)}")
        print(f"   ⏱️  Duración: {sesion.timestamp_fin - sesion.timestamp_inicio}")
        
        # Generar recomendaciones finales
        resultado = self.ml_predictor.generar_recomendaciones_completas(
            sesion.respuestas, top_n=5
        )
        
        # Agregar metadatos de la sesión
        resultado['session_metadata'] = {
            'session_id': session_id,
            'duracion_minutos': int((sesion.timestamp_fin - sesion.timestamp_inicio).total_seconds() / 60),
            'eficiencia_preguntas': f"{len(sesion.respuestas)}/{len(self.preguntas_fase1) + sum(len(b) for b in self.preguntas_fase2.values())}",
            'fase_final': sesion.fase_actual,
            'rama_detectada': sesion.rama_detectada,
            'categoria_detectada': sesion.categoria_detectada
        }
        
        print(f"✅ Recomendaciones generadas exitosamente")
        return resultado
    
    def obtener_estadisticas_sistema(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de cuestionario"""
        
        total_preguntas = len(self.preguntas_fase1)
        for banco in self.preguntas_fase2.values():
            total_preguntas += len(banco)
        
        return {
            'total_preguntas_disponibles': total_preguntas,
            'preguntas_fase1': len(self.preguntas_fase1),
            'preguntas_fase2_por_rama': {rama: len(banco) for rama, banco in self.preguntas_fase2.items()},
            'sesiones_activas': len(self.sesiones),
            'configuracion': self.config,
            'version': '2.0'
        }


def test_adaptive_questionnaire():
    """Prueba del sistema de cuestionario adaptativo"""
    print("🧪 PROBANDO ADAPTIVE QUESTIONNAIRE")
    print("=" * 50)
    
    try:
        # Inicializar predictor ML
        ml_predictor = VocationalMLPredictor(
            modelo_capacidad_path='/mnt/user-data/outputs/modelo_capacidad.pkl',
            modelo_rama_path='/mnt/user-data/outputs/modelo_rama.pkl',
            modelo_opciones_path='/mnt/user-data/outputs/modelo_opciones.pkl',
            opciones_info_path='/mnt/user-data/outputs/opciones_vocacionales.json'
        )
        
        # Inicializar cuestionario
        questionnaire = AdaptiveQuestionnaire(
            ml_predictor=ml_predictor,
            preguntas_fase1_path='/mnt/user-data/outputs/preguntas-fase1.json',
            preguntas_fase2_paths={
                'Salud': '/mnt/user-data/outputs/rama1_salud_preguntas_fase2.json',
                'Ingenieria': '/mnt/user-data/outputs/rama2_ingenieria_preguntas_fase2.json',
                'Negocios': '/mnt/user-data/outputs/rama3_negocios_preguntas_fase2.json',
                'Oficios_Tecnicos': '/mnt/user-data/outputs/rama4_oficios_tecnicos_preguntas_fase2.json',
                'Oficios_Basicos': '/mnt/user-data/outputs/rama5_oficios_basicos_preguntas_fase2.json'
            }
        )
        
        # Iniciar sesión
        sesion = questionnaire.iniciar_sesion()
        print(f"\n✅ Sesión iniciada: {sesion.session_id}")
        
        # Simular cuestionario
        for i in range(60):  # Simular hasta 60 preguntas
            
            # Obtener siguiente pregunta
            pregunta_data = questionnaire.obtener_siguiente_pregunta(sesion.session_id)
            
            if not pregunta_data:
                print(f"\n✅ Cuestionario terminado automáticamente en pregunta {i}")
                break
            
            pregunta = pregunta_data['pregunta']
            metadata = pregunta_data['metadata']
            
            # Simular respuesta (alternar A/B con algo de variación)
            if i < 30:
                respuesta = 'A' if i % 3 == 0 else 'B'  # Perfil técnico
            else:
                respuesta = random.choice(['A', 'B', 'C'])  # Más variación
            
            # Enviar respuesta
            resultado = questionnaire.enviar_respuesta(
                sesion.session_id, pregunta['id'], respuesta
            )
            
            # Mostrar progreso cada 10 preguntas
            if (i + 1) % 10 == 0:
                print(f"\n📊 Pregunta {i+1}: {pregunta['id']} → {respuesta}")
                print(f"   Fase: {metadata['fase']}")
                print(f"   Progreso: {resultado['progreso']['porcentaje_estimado']}%")
                
                if resultado.get('prediccion_parcial'):
                    top = resultado['prediccion_parcial']['top_3'][0]
                    print(f"   Top actual: {top[0]} ({top[1]})")
            
            if not resultado['debe_continuar']:
                print(f"\n✅ Sistema decidió terminar: {resultado['razon_estado']}")
                break
        
        # Finalizar y obtener resultados
        if len(sesion.respuestas) >= questionnaire.config['min_total_preguntas']:
            resultados = questionnaire.finalizar_cuestionario(sesion.session_id)
            
            print(f"\n🎯 RESULTADOS FINALES:")
            print(f"Capacidad: {resultados['capacidad_academica']['score']} ({resultados['capacidad_academica']['categoria']})")
            print(f"Total respuestas: {len(sesion.respuestas)}")
            
            print(f"\nTop 3 recomendaciones:")
            for rec in resultados['recomendaciones'][:3]:
                print(f"   {rec['ranking']}. {rec['nombre']} - {rec['match_score_porcentaje']}")
        
        print(f"\n✅ Prueba completada exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_adaptive_questionnaire()
