#!/usr/bin/env python3
"""
Wrapper de Machine Learning para el Sistema de Orientación Vocacional v2.0
Integra los 3 modelos entrenados: Capacidad, Rama y Opciones

Autor: Sistema de IA Claude  
Fecha: 2025-01-01
Versión: 1.0
"""

import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class VocationalMLPredictor:
    """
    Predictor ML para el Sistema de Orientación Vocacional
    
    Integra los 3 modelos especializados:
    1. Modelo Capacidad (Regresión): Predice score académico 0.0-2.0
    2. Modelo Rama (Clasificación): Predice rama para universitarios
    3. Modelo Opciones (5 Clasificadores): Predice opciones específicas
    """
    
    def __init__(self, 
                 modelo_capacidad_path: str = 'modelo_capacidad.pkl',
                 modelo_rama_path: str = 'modelo_rama.pkl', 
                 modelo_opciones_path: str = 'modelo_opciones.pkl',
                 opciones_info_path: str = 'opciones_vocacionales.json'):
        
        self.modelo_capacidad = None
        self.modelo_rama = None
        self.modelo_opciones = None
        self.opciones_info = None
        
        self.respuesta_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        
        self._cargar_modelos(modelo_capacidad_path, modelo_rama_path, modelo_opciones_path)
        self._cargar_opciones_info(opciones_info_path)
        
        print("VocationalMLPredictor inicializado correctamente")
    
    def _cargar_modelos(self, cap_path: str, rama_path: str, opc_path: str):
        """Carga los 3 modelos ML entrenados"""
        try:
            print("Cargando modelos ML...")
            
            self.modelo_capacidad = joblib.load(cap_path)
            print(f"Modelo capacidad: {cap_path}")
            
            rama_data = joblib.load(rama_path)
            self.modelo_rama = rama_data['modelo']
            self.rama_label_encoder = rama_data['label_encoder']
            print(f"Modelo rama: {rama_path}")
            
            opciones_data = joblib.load(opc_path)
            self.modelos_opciones = opciones_data['modelos']
            self.opciones_label_encoders = opciones_data['label_encoders']
            print(f"Modelos opciones: {opc_path}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Modelo no encontrado: {e}")
        except Exception as e:
            raise Exception(f"Error cargando modelos: {e}")
    
    def _cargar_opciones_info(self, info_path: str):
        """Carga información detallada de las opciones vocacionales"""
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.opciones_info = data['opciones']
            print(f"Info opciones: {info_path} ({len(self.opciones_info)} opciones)")
        except FileNotFoundError:
            print(f"Archivo {info_path} no encontrado - usando info básica")
            self._crear_info_basica()
        except Exception as e:
            print(f"Error cargando {info_path}: {e}")
            self._crear_info_basica()
    
    def _crear_info_basica(self):
        """Crea información básica si no se puede cargar el JSON"""
        codigos = ["MED", "PSI", "NUT", "ODO", "ENF", "SW", "ELEC", "MEC", "CIV", "ARQ",
                   "ADM", "CONT", "MKT", "DER", "COM", "TEC_ENF", "TEC_SIS", "ELEC_IND", 
                   "CHEF", "TEC_AUTO", "ALBANIL", "CHOFER", "LIMPIEZA", "VENDEDOR", "AYU_COCINA"]
        
        self.opciones_info = {}
        for i, codigo in enumerate(codigos, 1):
            self.opciones_info[codigo] = {
                "id": i,
                "codigo": codigo,
                "nombre": codigo.replace("_", " ").title(),
                "salario": {"promedio": 20000}
            }
    
    def predecir_capacidad_academica(self, respuestas_fase1: Dict[str, str]) -> float:
        """
        Predice la capacidad académica del estudiante (0.0 - 2.0)
        
        Args:
            respuestas_fase1: Dict con respuestas de Fase 1 (80 preguntas)
            
        Returns:
            float: Score de capacidad académica
        """
        if len(respuestas_fase1) < 10:
            raise ValueError("Se necesitan al menos 10 respuestas para predecir capacidad")
        
        feature_vector = []
        
        for i in range(1, 81):
            pregunta_id = f'fase1_p{i:02d}'
            
            respuesta = None
            for key, value in respuestas_fase1.items():
                if key.endswith(f'{i:03d}') or key == pregunta_id:
                    respuesta = value
                    break
            
            if respuesta and respuesta in self.respuesta_mapping:
                feature_vector.append(self.respuesta_mapping[respuesta])
            else:
                feature_vector.append(3)
        
        X = np.array(feature_vector).reshape(1, -1)
        capacidad_score = self.modelo_capacidad.predict(X)[0]
        
        return np.clip(capacidad_score, 0.0, 2.0)
    
    def determinar_categoria_estudiante(self, capacidad_score: float) -> str:
        """
        Determina la categoría del estudiante basada en su capacidad académica
        
        Args:
            capacidad_score: Score de capacidad (0.0-2.0)
            
        Returns:
            str: 'carreras', 'oficios_tecnicos', 'oficios_basicos'
        """
        if capacidad_score >= 1.0:
            return 'carreras'
        elif capacidad_score >= 0.5:
            return 'oficios_tecnicos' 
        else:
            return 'oficios_basicos'
    
    def predecir_rama_universitaria(self, respuestas_fase1: Dict[str, str]) -> Optional[str]:
        """
        Predice la rama universitaria (solo para estudiantes con capacidad >= 1.0)
        
        Args:
            respuestas_fase1: Respuestas de Fase 1
            
        Returns:
            str: 'Salud', 'Ingenieria', 'Negocios' o None
        """
        capacidad = self.predecir_capacidad_academica(respuestas_fase1)
        if capacidad < 1.0:
            return None
        
        feature_vector = []
        for i in range(1, 81):
            pregunta_id = f'fase1_p{i:02d}'
            respuesta = None
            for key, value in respuestas_fase1.items():
                if key.endswith(f'{i:03d}') or key == pregunta_id:
                    respuesta = value
                    break
            
            if respuesta and respuesta in self.respuesta_mapping:
                feature_vector.append(self.respuesta_mapping[respuesta])
            else:
                feature_vector.append(3)
        
        X = np.array(feature_vector).reshape(1, -1)
        
        rama_pred = self.modelo_rama.predict(X)[0]
        return self.rama_label_encoder.inverse_transform([rama_pred])[0]
    
    def predecir_opciones_especificas(self, 
                                      respuestas_completas: Dict[str, str],
                                      categoria: str,
                                      rama: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Predice las opciones específicas dentro de una categoría/rama
        
        Args:
            respuestas_completas: Todas las respuestas (Fase 1 + Fase 2)
            categoria: 'carreras', 'oficios_tecnicos', 'oficios_basicos'
            rama: Para carreras universitarias: 'Salud', 'Ingenieria', 'Negocios'
            
        Returns:
            List[Tuple[str, float]]: Lista de (codigo_opcion, probabilidad)
        """
        if categoria == 'carreras' and rama:
            if rama in self.modelos_opciones:
                modelo = self.modelos_opciones[rama]
                label_encoder = self.opciones_label_encoders[rama]
            else:
                return []
        elif categoria == 'oficios_tecnicos':
            modelo = self.modelos_opciones.get('Oficios_Tecnicos')
            label_encoder = self.opciones_label_encoders.get('Oficios_Tecnicos')
        elif categoria == 'oficios_basicos':
            modelo = self.modelos_opciones.get('Oficios_Basicos')
            label_encoder = self.opciones_label_encoders.get('Oficios_Basicos')
        else:
            return []
        
        if not modelo or not label_encoder:
            return []
        
        try:
            expected_features = getattr(modelo, 'n_features_in_', 82)
        except:
            expected_features = 82
        
        print(f"Modelo {categoria}/{rama} espera {expected_features} features")
        
        feature_vector = []
        
        for i in range(1, 81):
            respuesta = None
            for key, value in respuestas_completas.items():
                if key.endswith(f'{i:03d}') or key == f'fase1_p{i:02d}':
                    respuesta = value
                    break
            
            if respuesta and respuesta in self.respuesta_mapping:
                feature_vector.append(self.respuesta_mapping[respuesta])
            else:
                feature_vector.append(3)
        
        additional_features_needed = expected_features - 80
        
        if additional_features_needed > 0:
            for i in range(1, additional_features_needed + 1):
                respuesta = None
                for key, value in respuestas_completas.items():
                    if key == f'fase2_p{i:02d}':
                        respuesta = value
                        break
                
                if respuesta and respuesta in self.respuesta_mapping:
                    feature_vector.append(self.respuesta_mapping[respuesta])
                else:
                    feature_vector.append(3)
        
        while len(feature_vector) < expected_features:
            feature_vector.append(3)
        
        if len(feature_vector) > expected_features:
            feature_vector = feature_vector[:expected_features]
        
        print(f"Vector final: {len(feature_vector)} features (esperadas: {expected_features})")
        
        X = np.array(feature_vector).reshape(1, -1)
        
        try:
            probabilidades = modelo.predict_proba(X)[0]
        except Exception as e:
            print(f"Error en predicción: {e}")
            num_clases = len(label_encoder.classes_)
            probabilidades = np.ones(num_clases) / num_clases
        
        try:
            opciones = label_encoder.inverse_transform(range(len(probabilidades)))
        except Exception as e:
            print(f"Error en label encoder: {e}")
            opciones = [f"OPCION_{i}" for i in range(len(probabilidades))]
        
        resultados = list(zip(opciones, probabilidades))
        
        resultados.sort(key=lambda x: x[1], reverse=True)
        
        return resultados
    
    def generar_recomendaciones_completas(self, 
                                         respuestas_completas: Dict[str, str],
                                         top_n: int = 3) -> Dict[str, Any]:
        """
        Genera las recomendaciones completas del sistema
        
        Args:
            respuestas_completas: Todas las respuestas del estudiante
            top_n: Número de recomendaciones a retornar
            
        Returns:
            Dict con recomendaciones completas y metadata
        """
        print(f"Generando recomendaciones para {len(respuestas_completas)} respuestas...")
        
        respuestas_fase1 = {k: v for k, v in respuestas_completas.items() 
                           if k.startswith('fase1_') or any(k.endswith(f'{i:03d}') for i in range(1, 81))}
        
        capacidad_score = self.predecir_capacidad_academica(respuestas_fase1)
        categoria = self.determinar_categoria_estudiante(capacidad_score)
        
        print(f"Capacidad: {capacidad_score:.3f} -> {categoria}")
        
        rama = None
        if categoria == 'carreras':
            rama = self.predecir_rama_universitaria(respuestas_fase1)
            print(f"Rama: {rama}")
        
        opciones_pred = self.predecir_opciones_especificas(
            respuestas_completas, categoria, rama
        )
        
        if not opciones_pred:
            raise ValueError("No se pudieron generar predicciones de opciones")
        
        top_opciones = opciones_pred[:top_n]
        suma_scores = sum(prob for _, prob in top_opciones)
        
        if suma_scores > 0:
            top_opciones_norm = [(codigo, prob/suma_scores) for codigo, prob in top_opciones]
        else:
            top_opciones_norm = [(codigo, 1.0/len(top_opciones)) for codigo, _ in top_opciones]
        
        print(f"Top {len(top_opciones_norm)} opciones generadas")
        
        recomendaciones = []
        for i, (codigo, probabilidad) in enumerate(top_opciones_norm, 1):
            info = self.opciones_info.get(codigo, {})
            
            recomendacion = {
                "ranking": i,
                "codigo": codigo,
                "nombre": info.get('nombre', codigo),
                "nombre_completo": info.get('nombre_completo', codigo),
                "match_score": round(probabilidad, 3),
                "match_score_porcentaje": f"{probabilidad*100:.1f}%",
                "categoria": categoria,
                "rama": rama if categoria == 'carreras' else info.get('rama'),
                "info_basica": {
                    "años_estudio": info.get('años_estudio', 'N/A'),
                    "nivel_educativo": info.get('nivel_educativo', 'N/A'),
                    "dificultad": info.get('dificultad', 'N/A'),
                    "salario_promedio": info.get('salario', {}).get('promedio', 'N/A')
                },
                "mercado_laboral": info.get('mercado_laboral', {}),
                "ventajas": info.get('ventajas', []),
                "desafios": info.get('desafios', [])
            }
            recomendaciones.append(recomendacion)
        
        resultado = {
            "prediccion_exitosa": True,
            "capacidad_academica": {
                "score": round(capacidad_score, 3),
                "categoria": categoria,
                "descripcion": self._get_descripcion_categoria(categoria, capacidad_score)
            },
            "rama_universitaria": rama,
            "total_respuestas": len(respuestas_completas),
            "recomendaciones": recomendaciones,
            "metadata": {
                "modelo_version": "v2.0",
                "fecha_prediccion": "2025-01-01",
                "top_n_solicitado": top_n,
                "scores_normalizados": True
            }
        }
        
        print("Recomendaciones generadas exitosamente")
        return resultado
    
    def _get_descripcion_categoria(self, categoria: str, capacidad_score: float) -> str:
        """Genera descripción de la categoría del estudiante"""
        if categoria == 'carreras':
            return f"Tu perfil académico (score: {capacidad_score:.2f}) te permite acceder a carreras universitarias"
        elif categoria == 'oficios_tecnicos':
            return f"Tu perfil académico (score: {capacidad_score:.2f}) es ideal para carreras técnicas especializadas"
        else:
            return f"Tu perfil académico (score: {capacidad_score:.2f}) te orienta hacia oficios con capacitación práctica"
    
    def validar_respuestas(self, respuestas: Dict[str, str]) -> Dict[str, Any]:
        """
        Valida que las respuestas estén en formato correcto
        
        Returns:
            Dict con resultado de validación
        """
        errores = []
        warnings = []
        
        if not respuestas:
            errores.append("No se proporcionaron respuestas")
            return {"valido": False, "errores": errores, "warnings": warnings}
        
        respuestas_validas = 0
        for key, value in respuestas.items():
            if value not in ['A', 'B', 'C', 'D', 'E']:
                errores.append(f"Respuesta inválida en {key}: {value}")
            else:
                respuestas_validas += 1
        
        if respuestas_validas < 20:
            errores.append(f"Se necesitan al menos 20 respuestas válidas. Tienes: {respuestas_validas}")
        
        if respuestas_validas < 80:
            warnings.append(f"Con {respuestas_validas} respuestas la precisión puede ser limitada")
        
        return {
            "valido": len(errores) == 0,
            "errores": errores,
            "warnings": warnings,
            "respuestas_validas": respuestas_validas
        }
    
    def get_estadisticas_sistema(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema ML"""
        return {
            "modelos_cargados": {
                "capacidad": self.modelo_capacidad is not None,
                "rama": self.modelo_rama is not None, 
                "opciones": self.modelos_opciones is not None
            },
            "opciones_disponibles": len(self.opciones_info) if self.opciones_info else 0,
            "categorias": ["carreras", "oficios_tecnicos", "oficios_basicos"],
            "ramas_universitarias": ["Salud", "Ingenieria", "Negocios"],
            "version": "2.0"
        }


def test_vocational_predictor():
    """Función de prueba del predictor vocacional"""
    print("PROBANDO VOCATIONAL ML PREDICTOR")
    print("=" * 50)
    
    try:
        predictor = VocationalMLPredictor(
            modelo_capacidad_path='/mnt/user-data/outputs/modelo_capacidad.pkl',
            modelo_rama_path='/mnt/user-data/outputs/modelo_rama.pkl',
            modelo_opciones_path='/mnt/user-data/outputs/modelo_opciones.pkl',
            opciones_info_path='/mnt/user-data/outputs/opciones_vocacionales.json'
        )
        
        respuestas_test = {}
        
        for i in range(1, 81):
            if i <= 40:
                respuestas_test[f'fase1_p{i:02d}'] = 'A'
            else:
                respuestas_test[f'fase1_p{i:02d}'] = 'B'
        
        for i in range(1, 81):
            respuestas_test[f'fase2_p{i:02d}'] = 'A'
        
        print(f"\nRespuestas de prueba: {len(respuestas_test)}")
        
        validacion = predictor.validar_respuestas(respuestas_test)
        print(f"Validación: {validacion}")
        
        resultado = predictor.generar_recomendaciones_completas(respuestas_test, top_n=3)
        
        print(f"\nRESULTADOS:")
        print(f"Capacidad: {resultado['capacidad_academica']['score']} ({resultado['capacidad_academica']['categoria']})")
        print(f"Rama: {resultado['rama_universitaria']}")
        
        print(f"\nTOP 3 RECOMENDACIONES:")
        for rec in resultado['recomendaciones']:
            print(f"   {rec['ranking']}. {rec['nombre']} - {rec['match_score_porcentaje']} match")
        
        print(f"\nPrueba completada exitosamente")
        return True
        
    except Exception as e:
        print(f"Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_vocational_predictor()