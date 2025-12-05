#!/usr/bin/env python3
"""
Integración con LLM (Groq API) para generar explicaciones personalizadas
del Sistema de Orientación Vocacional v2.0

Autor: Sistema de IA Claude
Fecha: 2025-01-01
Versión: 1.0
"""

import json
import requests
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

class VocationalLLMExplainer:
    """
    Generador de explicaciones personalizadas usando Groq API (GRATIS)
    """
    
    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente LLM
        
        Args:
            api_key: API key de Groq (si no se provee, busca en variable de entorno)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            print("⚠️  ADVERTENCIA: No se encontró GROQ_API_KEY")
            print("   Las explicaciones serán básicas sin personalización")
            print("   Obtén una key gratis en: https://console.groq.com/keys")
            self.cliente_activo = False
        else:
            self.cliente_activo = True
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "llama-3.1-8b-instant"
            print("✅ VocationalLLMExplainer inicializado (Groq GRATIS)")
    
    def generar_reporte_completo(self, 
                                recomendaciones: List[Dict[str, Any]],
                                metadata_estudiante: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un reporte completo con explicaciones personalizadas
        
        Args:
            recomendaciones: Lista de recomendaciones del ML
            metadata_estudiante: Info del estudiante (capacidad, categoría, etc.)
            
        Returns:
            Dict con reporte completo enriquecido
        """
        print("📝 Generando reporte completo con explicaciones personalizadas...")
        
        if not self.cliente_activo:
            return self._generar_reporte_basico(recomendaciones, metadata_estudiante)
        
        # Enriquecer cada recomendación
        recomendaciones_enriquecidas = []
        
        for i, rec in enumerate(recomendaciones, 1):
            print(f"   Procesando {i}/{len(recomendaciones)}: {rec['nombre']}")
            
            explicacion = self._generar_explicacion_individual(
                recomendacion=rec,
                ranking=i,
                metadata_estudiante=metadata_estudiante,
                contexto_otras=recomendaciones
            )
            
            # Combinar datos originales con explicación LLM
            rec_enriquecida = rec.copy()
            rec_enriquecida.update(explicacion)
            recomendaciones_enriquecidas.append(rec_enriquecida)
            
            # Rate limiting - Groq es gratis pero limitado
            if i < len(recomendaciones):
                time.sleep(0.5)
        
        # Generar resumen ejecutivo
        resumen_ejecutivo = self._generar_resumen_ejecutivo(
            recomendaciones_enriquecidas, metadata_estudiante
        )
        
        # Generar mensaje motivacional
        mensaje_motivacional = self._generar_mensaje_motivacional(
            metadata_estudiante, recomendaciones_enriquecidas[0]
        )
        
        return {
            "fecha_evaluacion": datetime.now().isoformat(),
            "resumen_ejecutivo": resumen_ejecutivo,
            "mensaje_motivacional": mensaje_motivacional,
            "recomendaciones": recomendaciones_enriquecidas,
            "metadata": {
                "total_opciones_evaluadas": len(recomendaciones),
                "capacidad_academica": metadata_estudiante.get('capacidad_academica', {}),
                "categoria_estudiante": metadata_estudiante.get('categoria'),
                "rama_universitaria": metadata_estudiante.get('rama_universitaria'),
                "modelo_llm": "Groq Llama 3.1 8B (GRATIS)" if self.cliente_activo else "Básico",
                "version_sistema": "2.0",
                "tiempo_generacion": "~3-5 segundos"
            }
        }
    
    def _generar_explicacion_individual(self, 
                                       recomendacion: Dict[str, Any],
                                       ranking: int,
                                       metadata_estudiante: Dict[str, Any],
                                       contexto_otras: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera explicación personalizada para una recomendación específica"""
        
        prompt = self._construir_prompt_individual(
            recomendacion, ranking, metadata_estudiante, contexto_otras
        )
        
        try:
            respuesta = self._llamar_groq_api(prompt, max_tokens=400)
            
            if respuesta:
                return {
                    "explicacion_personalizada": respuesta,
                    "explicacion_generada": True
                }
            else:
                return self._explicacion_fallback(recomendacion, ranking)
                
        except Exception as e:
            print(f"⚠️  Error generando explicación para {recomendacion['nombre']}: {e}")
            return self._explicacion_fallback(recomendacion, ranking)
    
    def _construir_prompt_individual(self, 
                                   recomendacion: Dict[str, Any], 
                                   ranking: int,
                                   metadata_estudiante: Dict[str, Any],
                                   contexto_otras: List[Dict[str, Any]]) -> str:
        """Construye el prompt para explicación individual"""
        
        capacidad = metadata_estudiante.get('capacidad_academica', {})
        categoria = metadata_estudiante.get('categoria', 'N/A')
        
        # Información de la recomendación
        nombre = recomendacion['nombre']
        match_score = recomendacion['match_score_porcentaje']
        info_basica = recomendacion.get('info_basica', {})
        
        # Construir contexto de comparación
        if len(contexto_otras) > 1:
            otras_opciones = [r['nombre'] for r in contexto_otras[:3] if r['nombre'] != nombre]
            contexto_comp = f"Entre tus top opciones también están: {', '.join(otras_opciones[:2])}"
        else:
            contexto_comp = "Esta es tu opción más compatible"
        
        prompt = f"""Eres un orientador vocacional mexicano experto ayudando a un estudiante a entender sus resultados.

CONTEXTO DEL ESTUDIANTE:
- Capacidad académica: {capacidad.get('score', 'N/A')} ({capacidad.get('categoria', 'N/A')})
- Perfil: {categoria}
- Resultado #{ranking} de sus recomendaciones

OPCIÓN RECOMENDADA:
- Carrera/Oficio: {nombre}
- Compatibilidad: {match_score}
- Duración: {info_basica.get('años_estudio', 'N/A')} años
- Salario promedio: ${info_basica.get('salario_promedio', 'N/A'):,} MXN
{contexto_comp}

TAREA:
Escribe una explicación personalizada de 120-150 palabras que:

1. Explique por qué {nombre} es una excelente opción para este estudiante específico
2. Destaque 2-3 aspectos que hacen esta opción especialmente adecuada
3. Mencione oportunidades realistas en México 2025
4. Incluya un mensaje motivador pero realista

ESTILO:
- Tono cercano y motivador (usa "tú")
- Enfoque en beneficios específicos del estudiante
- Menciona contexto mexicano actual (nearshoring, demanda, etc.)
- Sin bullet points, párrafos fluidos
- Honesto pero optimista"""

        return prompt
    
    def _generar_resumen_ejecutivo(self, 
                                  recomendaciones: List[Dict[str, Any]],
                                  metadata_estudiante: Dict[str, Any]) -> str:
        """Genera resumen ejecutivo personalizado"""
        
        if not self.cliente_activo:
            return self._resumen_ejecutivo_basico(recomendaciones, metadata_estudiante)
        
        top_3 = [r['nombre'] for r in recomendaciones[:3]]
        capacidad = metadata_estudiante.get('capacidad_academica', {})
        categoria = metadata_estudiante.get('categoria', 'estudiante')
        
        prompt = f"""Eres un orientador vocacional escribiendo un resumen ejecutivo para un estudiante mexicano.

PERFIL DEL ESTUDIANTE:
- Capacidad académica: {capacidad.get('score', 'N/A')} 
- Categoría: {categoria}
- Top 3 recomendaciones: {', '.join(top_3)}

TAREA:
Escribe un resumen ejecutivo de 80-100 palabras que:

1. Felicite al estudiante por completar la evaluación
2. Resuma su perfil de forma positiva
3. Identifique el patrón común en sus recomendaciones
4. Lo motive a explorar las opciones con confianza
5. Mencione el contexto de oportunidades en México

ESTILO:
- Profesional pero cálido
- Enfoque en fortalezas y oportunidades
- Mensaje motivador y realista
- Habla directamente al estudiante (tú)"""

        try:
            return self._llamar_groq_api(prompt, max_tokens=200) or \
                   self._resumen_ejecutivo_basico(recomendaciones, metadata_estudiante)
        except:
            return self._resumen_ejecutivo_basico(recomendaciones, metadata_estudiante)
    
    def _generar_mensaje_motivacional(self, 
                                    metadata_estudiante: Dict[str, Any],
                                    top_recomendacion: Dict[str, Any]) -> str:
        """Genera mensaje motivacional personalizado"""
        
        if not self.cliente_activo:
            return "¡Tu futuro profesional te espera! Cada una de estas opciones puede llevarte al éxito con dedicación y esfuerzo."
        
        categoria = metadata_estudiante.get('categoria', 'estudiante')
        top_opcion = top_recomendacion['nombre']
        
        prompt = f"""Escribe un mensaje motivacional de 40-50 palabras para un estudiante mexicano {categoria} cuya top recomendación es {top_opcion}.

El mensaje debe:
- Ser inspirador y realista
- Mencionar que tiene un buen futuro por delante
- Animarlo a tomar acción
- Ser específico a México 2025

Tono: Motivador, directo, optimista."""

        try:
            return self._llamar_groq_api(prompt, max_tokens=100) or \
                   f"¡Excelente! Tu perfil para {top_opcion} muestra gran potencial. México necesita profesionales como tú. ¡Es tu momento de brillar!"
        except:
            return f"¡Excelente! Tu perfil para {top_opcion} muestra gran potencial. México necesita profesionales como tú. ¡Es tu momento de brillar!"
    
    def _llamar_groq_api(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Realiza llamada a la API de Groq"""
        
        if not self.cliente_activo:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"⚠️  Error Groq API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("⚠️  Timeout en Groq API")
            return None
        except Exception as e:
            print(f"⚠️  Error inesperado Groq API: {e}")
            return None
    
    def _generar_reporte_basico(self, 
                               recomendaciones: List[Dict[str, Any]], 
                               metadata_estudiante: Dict[str, Any]) -> Dict[str, Any]:
        """Genera reporte básico sin LLM"""
        
        recomendaciones_basicas = []
        for i, rec in enumerate(recomendaciones, 1):
            rec_basica = rec.copy()
            rec_basica.update(self._explicacion_fallback(rec, i))
            recomendaciones_basicas.append(rec_basica)
        
        return {
            "fecha_evaluacion": datetime.now().isoformat(),
            "resumen_ejecutivo": self._resumen_ejecutivo_basico(recomendaciones, metadata_estudiante),
            "mensaje_motivacional": "¡Tu evaluación muestra un futuro prometedor! Explora estas opciones y decide con confianza.",
            "recomendaciones": recomendaciones_basicas,
            "metadata": {
                "total_opciones_evaluadas": len(recomendaciones),
                "capacidad_academica": metadata_estudiante.get('capacidad_academica', {}),
                "modelo_llm": "Básico (sin personalización)",
                "version_sistema": "2.0"
            }
        }
    
    def _explicacion_fallback(self, recomendacion: Dict[str, Any], ranking: int) -> Dict[str, Any]:
        """Explicación básica sin LLM"""
        
        nombre = recomendacion['nombre']
        match_score = recomendacion['match_score_porcentaje']
        info = recomendacion.get('info_basica', {})
        
        explicacion = f"Tu compatibilidad con {nombre} es de {match_score}, lo que indica una buena alineación con tu perfil. "
        
        if info.get('años_estudio'):
            explicacion += f"Esta opción requiere {info['años_estudio']} años de estudio. "
        
        if ranking == 1:
            explicacion += "Es tu mejor opción basada en tus respuestas y ofrece excelentes oportunidades en México."
        else:
            explicacion += "Es una alternativa sólida que vale la pena considerar para tu futuro profesional."
        
        return {
            "explicacion_personalizada": explicacion,
            "explicacion_generada": False
        }
    
    def _resumen_ejecutivo_basico(self, 
                                 recomendaciones: List[Dict[str, Any]], 
                                 metadata_estudiante: Dict[str, Any]) -> str:
        """Resumen ejecutivo básico sin LLM"""
        
        categoria = metadata_estudiante.get('categoria', 'estudiante')
        top_opcion = recomendaciones[0]['nombre'] if recomendaciones else "opción recomendada"
        
        return f"¡Felicidades por completar tu evaluación vocacional! Tu perfil como {categoria} muestra gran potencial, " \
               f"especialmente en {top_opcion}. Las opciones recomendadas están alineadas con tus fortalezas y " \
               f"ofrecen buenas oportunidades en el mercado mexicano actual. Explora cada opción con confianza."
    
    def validar_configuracion(self) -> Dict[str, Any]:
        """Valida la configuración del servicio LLM"""
        
        if not self.cliente_activo:
            return {
                "configurado": False,
                "servicio": "Básico",
                "mensaje": "Groq API no configurada - usando explicaciones básicas",
                "recomendacion": "Configura GROQ_API_KEY para explicaciones personalizadas"
            }
        
        # Probar conexión con Groq
        try:
            test_response = self._llamar_groq_api("Test de conexión", max_tokens=10)
            
            return {
                "configurado": True,
                "servicio": "Groq API",
                "modelo": self.model,
                "estado": "Operativo",
                "test_exitoso": test_response is not None,
                "mensaje": "Explicaciones personalizadas disponibles"
            }
            
        except Exception as e:
            return {
                "configurado": False,
                "servicio": "Groq API",
                "error": str(e),
                "mensaje": "Error de conexión - usando explicaciones básicas"
            }
    
    def get_estadisticas_uso(self) -> Dict[str, Any]:
        """Estadísticas de uso del servicio LLM"""
        
        return {
            "servicio_activo": self.cliente_activo,
            "modelo_utilizado": self.model if self.cliente_activo else "N/A",
            "costo_por_uso": "$0.00 (GRATIS)" if self.cliente_activo else "N/A",
            "limite_diario": "14,400 requests/día" if self.cliente_activo else "N/A",
            "tiempo_respuesta_promedio": "1-3 segundos" if self.cliente_activo else "N/A"
        }


def test_llm_explainer():
    """Prueba del sistema de explicaciones LLM"""
    print("🧪 PROBANDO VOCATIONAL LLM EXPLAINER")
    print("=" * 50)
    
    # Inicializar explainer
    explainer = VocationalLLMExplainer()
    
    # Validar configuración
    config = explainer.validar_configuracion()
    print(f"\n📋 Configuración: {config}")
    
    # Datos de prueba
    recomendaciones_test = [
        {
            "ranking": 1,
            "codigo": "SW",
            "nombre": "Ingeniería en Software",
            "match_score": 0.85,
            "match_score_porcentaje": "85.0%",
            "categoria": "carreras",
            "info_basica": {
                "años_estudio": 4,
                "salario_promedio": 45000,
                "nivel_educativo": "Licenciatura"
            }
        },
        {
            "ranking": 2,
            "codigo": "TEC_SIS", 
            "nombre": "Técnico en Sistemas",
            "match_score": 0.78,
            "match_score_porcentaje": "78.0%",
            "categoria": "oficios_tecnicos",
            "info_basica": {
                "años_estudio": 3,
                "salario_promedio": 22000,
                "nivel_educativo": "Técnico Superior"
            }
        }
    ]
    
    metadata_test = {
        "capacidad_academica": {
            "score": 1.45,
            "categoria": "carreras"
        },
        "categoria": "universitario",
        "rama_universitaria": "Ingenieria"
    }
    
    print(f"\n🔄 Generando reporte de prueba...")
    
    # Generar reporte
    reporte = explainer.generar_reporte_completo(
        recomendaciones=recomendaciones_test,
        metadata_estudiante=metadata_test
    )
    
    # Mostrar resultados
    print(f"\n📄 REPORTE GENERADO:")
    print("=" * 50)
    
    print(f"\n📝 Resumen Ejecutivo:")
    print(f"   {reporte['resumen_ejecutivo']}")
    
    print(f"\n💪 Mensaje Motivacional:")
    print(f"   {reporte['mensaje_motivacional']}")
    
    print(f"\n🎯 Explicaciones Personalizadas:")
    for rec in reporte['recomendaciones']:
        print(f"\n   {rec['ranking']}. {rec['nombre']} ({rec['match_score_porcentaje']})")
        print(f"      {rec['explicacion_personalizada'][:100]}...")
    
    print(f"\n📊 Metadata:")
    for key, value in reporte['metadata'].items():
        print(f"   {key}: {value}")
    
    # Estadísticas de uso
    stats = explainer.get_estadisticas_uso()
    print(f"\n📈 Estadísticas de Uso:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Prueba completada exitosamente!")
    return True


if __name__ == "__main__":
    test_llm_explainer()
