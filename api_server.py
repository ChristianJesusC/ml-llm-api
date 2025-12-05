from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import uvicorn
import json
import os
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# Imports de nuestros módulos
from ml_model import VocationalMLPredictor
from adaptive_questionnaire import AdaptiveQuestionnaire, QuestionnaireSession
from llm_integration_groq import VocationalLLMExplainer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic para la API
class SessionStartRequest(BaseModel):
    """Request para iniciar sesión"""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SessionStartResponse(BaseModel):
    """Response al iniciar sesión"""
    session_id: str
    message: str
    timestamp: str

class AnswerRequest(BaseModel):
    """Request para enviar respuesta"""
    pregunta_id: str
    respuesta: str = Field(pattern="^[ABCDE]$")
    
    @validator('respuesta')
    def validate_respuesta(cls, v):
        if v not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError('Respuesta debe ser A, B, C, D o E')
        return v

class NextQuestionResponse(BaseModel):
    """Response con siguiente pregunta"""
    pregunta: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    session_status: str

class AnswerResponse(BaseModel):
    """Response al enviar respuesta"""
    respuesta_registrada: bool
    total_respuestas: int
    progreso: Dict[str, Any]
    debe_continuar: bool
    puede_finalizar: bool
    prediccion_parcial: Optional[Dict[str, Any]]

class FinishResponse(BaseModel):
    """Response al finalizar cuestionario"""
    reporte_completo: Dict[str, Any]
    session_summary: Dict[str, Any]

class SystemStatsResponse(BaseModel):
    """Response con estadísticas del sistema"""
    ml_system: Dict[str, Any]
    questionnaire_system: Dict[str, Any]
    llm_system: Dict[str, Any]
    active_sessions: int

# Variables globales del sistema
system_components = {
    'ml_predictor': None,
    'questionnaire': None,
    'llm_explainer': None,
    'active_sessions': {}
}

# Configuración del sistema
CONFIG = {
    'models_path': 'outputs',
    'max_active_sessions': 100,
    'session_timeout_hours': 24,
    'enable_cors': True,
    'enable_llm': True,
    'groq_api_key': os.getenv('GROQ_API_KEY')
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("🚀 Inicializando Sistema de Orientación Vocacional v2.0...")
    
    try:
        # Inicializar componentes ML
        await initialize_system_components()
        logger.info("✅ Todos los componentes inicializados correctamente")
        
        # Configurar limpieza de sesiones
        asyncio.create_task(cleanup_expired_sessions())
        
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando sistema...")
    system_components['active_sessions'].clear()
    logger.info("✅ Sistema cerrado correctamente")

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Orientación Vocacional v2.0",
    description="API para orientación vocacional con ML adaptativo y explicaciones personalizadas",
    version="2.0.0",
    lifespan=lifespan
)

# Configurar CORS
if CONFIG['enable_cors']:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar dominios exactos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Montar archivos estáticos (para frontend)
# app.mount("/static", StaticFiles(directory="static"), name="static")

async def initialize_system_components():
    """Inicializa todos los componentes del sistema"""
    
    models_path = CONFIG['models_path']
    
    # 1. Inicializar ML Predictor
    logger.info("📊 Inicializando ML Predictor...")
    system_components['ml_predictor'] = VocationalMLPredictor(
        modelo_capacidad_path=f"{models_path}/modelo_capacidad.pkl",
        modelo_rama_path=f"{models_path}/modelo_rama.pkl", 
        modelo_opciones_path=f"{models_path}/modelo_opciones.pkl",
        opciones_info_path=f"{models_path}/opciones_vocacionales.json"
    )
    
    # 2. Inicializar Adaptive Questionnaire
    logger.info("📝 Inicializando Adaptive Questionnaire...")
    system_components['questionnaire'] = AdaptiveQuestionnaire(
        ml_predictor=system_components['ml_predictor'],
        preguntas_fase1_path=f"{models_path}/preguntas-fase1.json",
        preguntas_fase2_paths={
            'Salud': f"{models_path}/rama1_salud_preguntas_fase2.json",
            'Ingenieria': f"{models_path}/rama2_ingenieria_preguntas_fase2.json",
            'Negocios': f"{models_path}/rama3_negocios_preguntas_fase2.json",
            'Oficios_Tecnicos': f"{models_path}/rama4_oficios_tecnicos_preguntas_fase2.json",
            'Oficios_Basicos': f"{models_path}/rama5_oficios_basicos_preguntas_fase2.json"
        }
    )
    
    # 3. Inicializar LLM Explainer
    if CONFIG['enable_llm']:
        logger.info("🗣️ Inicializando LLM Explainer...")
        system_components['llm_explainer'] = VocationalLLMExplainer(
            api_key=CONFIG['groq_api_key']
        )
    else:
        logger.info("⚠️ LLM Explainer deshabilitado")
        system_components['llm_explainer'] = VocationalLLMExplainer(api_key=None)

def get_ml_predictor() -> VocationalMLPredictor:
    """Dependency injection para ML Predictor"""
    if system_components['ml_predictor'] is None:
        raise HTTPException(status_code=503, detail="ML Predictor no inicializado")
    return system_components['ml_predictor']

def get_questionnaire() -> AdaptiveQuestionnaire:
    """Dependency injection para Questionnaire"""
    if system_components['questionnaire'] is None:
        raise HTTPException(status_code=503, detail="Questionnaire no inicializado")
    return system_components['questionnaire']

def get_llm_explainer() -> VocationalLLMExplainer:
    """Dependency injection para LLM Explainer"""
    if system_components['llm_explainer'] is None:
        raise HTTPException(status_code=503, detail="LLM Explainer no inicializado")
    return system_components['llm_explainer']

async def cleanup_expired_sessions():
    """Limpia sesiones expiradas periódicamente"""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_info in system_components['active_sessions'].items():
                if 'timestamp' in session_info:
                    session_time = datetime.fromisoformat(session_info['timestamp'])
                    if current_time - session_time > timedelta(hours=CONFIG['session_timeout_hours']):
                        expired_sessions.append(session_id)
            
            # Remover sesiones expiradas
            for session_id in expired_sessions:
                del system_components['active_sessions'][session_id]
                logger.info(f"🧹 Sesión expirada removida: {session_id}")
            
            # Dormir por 1 hora antes de siguiente limpieza
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error en cleanup de sesiones: {e}")
            await asyncio.sleep(3600)

# ============================================
# ENDPOINTS DE LA API
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página de inicio con información del sistema"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Orientación Vocacional v2.0</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; }
            .post { background: #28a745; }
            .get { background: #007bff; }
            .stats { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎓 Sistema de Orientación Vocacional v2.0</h1>
            <p>API inteligente con Machine Learning y explicaciones personalizadas</p>
        </div>
        
        <div class="stats">
            <h3>📊 Estado del Sistema</h3>
            <p><strong>Estado:</strong> ✅ Operativo</p>
            <p><strong>Versión:</strong> 2.0.0</p>
            <p><strong>Modelos ML:</strong> Capacidad, Rama, Opciones (3 modelos)</p>
            <p><strong>LLM:</strong> Groq Llama 3.1 8B (GRATIS)</p>
            <p><strong>Preguntas:</strong> 480 total (80 Fase 1 + 400 Fase 2)</p>
        </div>

        <h3>🛠️ Endpoints Disponibles</h3>
        
        <div class="endpoint">
            <span class="method post">POST</span> 
            <code>/api/v2/session/start</code><br>
            <small>Inicia nueva sesión de evaluación vocacional</small>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> 
            <code>/api/v2/session/{session_id}/next-question</code><br>
            <small>Obtiene siguiente pregunta del cuestionario adaptativo</small>
        </div>

        <div class="endpoint">
            <span class="method post">POST</span> 
            <code>/api/v2/session/{session_id}/answer</code><br>
            <small>Envía respuesta del estudiante (A, B, C, D, E)</small>
        </div>

        <div class="endpoint">
            <span class="method post">POST</span> 
            <code>/api/v2/session/{session_id}/finish</code><br>
            <small>Finaliza evaluación y genera recomendaciones completas</small>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> 
            <code>/api/v2/session/{session_id}/status</code><br>
            <small>Consulta estado actual de la sesión</small>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> 
            <code>/api/v2/system/stats</code><br>
            <small>Estadísticas y estado de todos los componentes</small>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> 
            <code>/api/v2/health</code><br>
            <small>Health check del sistema</small>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> 
            <code>/docs</code><br>
            <small>Documentación interactiva de la API (Swagger)</small>
        </div>

        <p style="text-align: center; margin-top: 40px; color: #6c757d;">
            <small>Sistema desarrollado con FastAPI, Scikit-learn y Groq LLM</small>
        </p>
    </body>
    </html>
    """

@app.get("/api/v2/health")
async def health_check():
    """Health check del sistema"""
    try:
        # Verificar componentes críticos
        ml_status = system_components['ml_predictor'] is not None
        questionnaire_status = system_components['questionnaire'] is not None
        llm_status = system_components['llm_explainer'] is not None
        
        all_healthy = ml_status and questionnaire_status and llm_status
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "ml_predictor": "ok" if ml_status else "error",
                "questionnaire": "ok" if questionnaire_status else "error", 
                "llm_explainer": "ok" if llm_status else "error"
            },
            "active_sessions": len(system_components['active_sessions'])
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="System health check failed")

@app.post("/api/v2/session/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    """Inicia una nueva sesión de evaluación vocacional"""
    try:
        # Verificar límite de sesiones activas
        if len(system_components['active_sessions']) >= CONFIG['max_active_sessions']:
            raise HTTPException(
                status_code=429, 
                detail=f"Máximo de {CONFIG['max_active_sessions']} sesiones activas alcanzado"
            )
        
        # Crear nueva sesión
        session = questionnaire.iniciar_sesion()
        
        # Registrar en sesiones activas
        system_components['active_sessions'][session.session_id] = {
            'session': session,
            'timestamp': datetime.now().isoformat(),
            'metadata': request.metadata
        }
        
        logger.info(f"📋 Nueva sesión iniciada: {session.session_id}")
        
        return SessionStartResponse(
            session_id=session.session_id,
            message="Sesión iniciada exitosamente",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error iniciando sesión: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/v2/session/{session_id}/next-question", response_model=NextQuestionResponse)
async def get_next_question(
    session_id: str,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    """Obtiene la siguiente pregunta del cuestionario adaptativo"""
    try:
        # Verificar que la sesión existe
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        # Obtener siguiente pregunta
        pregunta_data = questionnaire.obtener_siguiente_pregunta(session_id)
        
        if pregunta_data is None:
            # El cuestionario debe terminar
            return NextQuestionResponse(
                pregunta=None,
                metadata={"debe_terminar": True, "razon": "Cuestionario completo"},
                session_status="ready_to_finish"
            )
        
        return NextQuestionResponse(
            pregunta=pregunta_data['pregunta'],
            metadata=pregunta_data['metadata'],
            session_status="active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo siguiente pregunta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/api/v2/session/{session_id}/answer", response_model=AnswerResponse)
async def submit_answer(
    session_id: str,
    answer: AnswerRequest,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    """Registra una respuesta del estudiante"""
    try:
        # Verificar que la sesión existe
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        # Enviar respuesta
        resultado = questionnaire.enviar_respuesta(
            session_id=session_id,
            pregunta_id=answer.pregunta_id,
            respuesta=answer.respuesta
        )
        
        # Actualizar timestamp de la sesión
        system_components['active_sessions'][session_id]['timestamp'] = datetime.now().isoformat()
        
        return AnswerResponse(
            respuesta_registrada=resultado['respuesta_registrada'],
            total_respuestas=resultado['total_respuestas'],
            progreso=resultado['progreso'],
            debe_continuar=resultado['debe_continuar'],
            puede_finalizar=resultado['puede_finalizar'],
            prediccion_parcial=resultado.get('prediccion_parcial')
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error procesando respuesta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/api/v2/session/{session_id}/finish", response_model=FinishResponse)
async def finish_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire),
    llm_explainer: VocationalLLMExplainer = Depends(get_llm_explainer)
):
    """Finaliza la evaluación y genera recomendaciones completas"""
    try:
        # Verificar que la sesión existe
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        session_info = system_components['active_sessions'][session_id]
        
        # Finalizar cuestionario y obtener recomendaciones
        resultado_ml = questionnaire.finalizar_cuestionario(session_id)
        
        # Generar explicaciones personalizadas con LLM
        logger.info(f"🗣️ Generando explicaciones LLM para sesión {session_id}")
        
        reporte_completo = llm_explainer.generar_reporte_completo(
            recomendaciones=resultado_ml['recomendaciones'],
            metadata_estudiante={
                'capacidad_academica': resultado_ml['capacidad_academica'],
                'categoria': resultado_ml['capacidad_academica']['categoria'],
                'rama_universitaria': resultado_ml.get('rama_universitaria')
            }
        )
        
        # Combinar resultados ML + LLM
        reporte_final = {
            **resultado_ml,
            **reporte_completo
        }
        
        # Summary de la sesión
        session_summary = {
            'session_id': session_id,
            'duracion_total': resultado_ml['session_metadata']['duracion_minutos'],
            'total_respuestas': resultado_ml['total_respuestas'],
            'eficiencia': resultado_ml['session_metadata']['eficiencia_preguntas'],
            'categoria_final': resultado_ml['capacidad_academica']['categoria'],
            'top_recomendacion': resultado_ml['recomendaciones'][0]['nombre'],
            'timestamp_finalizacion': datetime.now().isoformat()
        }
        
        # Programar limpieza de sesión en background
        background_tasks.add_task(cleanup_session, session_id)
        
        logger.info(f"✅ Sesión {session_id} finalizada exitosamente")
        logger.info(f"   📊 {session_summary['total_respuestas']} respuestas en {session_summary['duracion_total']} min")
        logger.info(f"   🎯 Top: {session_summary['top_recomendacion']}")
        
        return FinishResponse(
            reporte_completo=reporte_final,
            session_summary=session_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finalizando sesión {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generando reporte final")

@app.get("/api/v2/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Obtiene el estado actual de una sesión"""
    try:
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        session_info = system_components['active_sessions'][session_id]
        session = session_info['session']
        
        return {
            'session_id': session_id,
            'status': 'active',
            'estadisticas': session.estadisticas,
            'fase_actual': session.fase_actual,
            'categoria_detectada': session.categoria_detectada,
            'rama_detectada': session.rama_detectada,
            'capacidad_estimada': session.capacidad_estimada,
            'timestamp_inicio': session.timestamp_inicio.isoformat(),
            'timestamp_ultima_actividad': session_info['timestamp'],
            'metadata_inicial': session_info.get('metadata', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo status de sesión: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/v2/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    ml_predictor: VocationalMLPredictor = Depends(get_ml_predictor),
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire),
    llm_explainer: VocationalLLMExplainer = Depends(get_llm_explainer)
):
    """Obtiene estadísticas completas del sistema"""
    try:
        return SystemStatsResponse(
            ml_system=ml_predictor.get_estadisticas_sistema(),
            questionnaire_system=questionnaire.obtener_estadisticas_sistema(),
            llm_system=llm_explainer.get_estadisticas_uso(),
            active_sessions=len(system_components['active_sessions'])
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

async def cleanup_session(session_id: str):
    """Limpia una sesión específica (background task)"""
    try:
        if session_id in system_components['active_sessions']:
            del system_components['active_sessions'][session_id]
            logger.info(f"🧹 Sesión {session_id} limpiada")
    except Exception as e:
        logger.error(f"Error limpiando sesión {session_id}: {e}")

# ============================================
# ENDPOINTS ADICIONALES DE UTILIDAD
# ============================================

@app.get("/api/v2/opciones")
async def get_opciones_vocacionales(
    ml_predictor: VocationalMLPredictor = Depends(get_ml_predictor)
):
    """Obtiene la lista completa de opciones vocacionales disponibles"""
    try:
        return {
            "opciones": ml_predictor.opciones_info,
            "total": len(ml_predictor.opciones_info),
            "categorias": {
                "carreras": [k for k, v in ml_predictor.opciones_info.items() if v.get('rama') in ['Salud', 'Ingenieria', 'Negocios']],
                "oficios_tecnicos": [k for k, v in ml_predictor.opciones_info.items() if v.get('rama') == 'Oficios_Tecnicos'],
                "oficios_basicos": [k for k, v in ml_predictor.opciones_info.items() if v.get('rama') == 'Oficios_Basicos']
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo opciones: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/v2/demo")
async def demo_prediction():
    """Endpoint de demostración con predicción de ejemplo"""
    try:
        # Respuestas de ejemplo para demo
        respuestas_demo = {}
        
        # Simular respuestas de un estudiante con perfil técnico
        for i in range(1, 61):  # 60 respuestas
            if i <= 30:
                respuestas_demo[f'fase1_p{i:02d}'] = 'A' if i % 3 == 0 else 'B'
            else:
                respuestas_demo[f'fase2_p{(i-30):02d}'] = 'A'
        
        # Obtener predictor y generar recomendaciones
        ml_predictor = get_ml_predictor()
        resultado = ml_predictor.generar_recomendaciones_completas(
            respuestas_demo, top_n=3
        )
        
        return {
            "demo": True,
            "mensaje": "Predicción de ejemplo con perfil técnico simulado",
            "respuestas_simuladas": len(respuestas_demo),
            "resultado": resultado
        }
        
    except Exception as e:
        logger.error(f"Error en demo: {e}")
        raise HTTPException(status_code=500, detail="Error generando demo")

# ============================================
# CONFIGURACIÓN DEL SERVIDOR
# ============================================

def create_app():
    """Factory para crear la aplicación"""
    return app

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info",
        access_log=True
    )