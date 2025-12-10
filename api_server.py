from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import uvicorn
import os
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import traceback

from ml_model import VocationalMLPredictor
from adaptive_questionnaire import AdaptiveQuestionnaire, QuestionnaireSession
from llm_integration_groq import VocationalLLMExplainer

logging.basicConfig(level=logging.INFO)
load_dotenv()
logger = logging.getLogger(__name__)

class SessionStartRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SessionStartResponse(BaseModel):
    session_id: str
    message: str
    timestamp: str

class AnswerRequest(BaseModel):
    pregunta_id: str
    respuesta: str = Field(pattern="^[ABCDE]$")
    
    @validator('respuesta')
    def validate_respuesta(cls, v):
        if v not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError('Respuesta debe ser A, B, C, D o E')
        return v

class NextQuestionResponse(BaseModel):
    pregunta: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    session_status: str

class AnswerResponse(BaseModel):
    respuesta_registrada: bool
    total_respuestas: int
    progreso: Dict[str, Any]
    debe_continuar: bool
    puede_finalizar: bool
    prediccion_parcial: Optional[Dict[str, Any]]

class FinishResponse(BaseModel):
    session_id: str
    capacidad_academica: Dict[str, Any]
    rama_universitaria: Optional[str]
    top_3_recomendaciones: List[Dict[str, Any]]
    resumen_ejecutivo: str
    mensaje_motivacional: str
    total_respuestas: int
    timestamp: str
    llm_disponible: bool

class SystemStatsResponse(BaseModel):
    ml_system: Dict[str, Any]
    questionnaire_system: Dict[str, Any]
    llm_system: Dict[str, Any]
    active_sessions: int

system_components = {
    'ml_predictor': None,
    'questionnaire': None,
    'llm_explainer': None,
    'active_sessions': {}
}

CONFIG = {
    'models_path': 'outputs',
    'max_active_sessions': 100,
    'session_timeout_hours': 24,
    'enable_cors': True,
    'enable_llm': True,
    'groq_api_key': os.getenv('GROQ_API_KEY'),
    'top_n_recomendaciones': 3
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.system('cls')
    logger.info("Inicializando Sistema de Orientacion Vocacional v2.0...")
    
    try:
        await initialize_system_components()
        logger.info("Todos los componentes inicializados correctamente")
        asyncio.create_task(cleanup_expired_sessions())
    except Exception as e:
        logger.error(f"Error inicializando sistema: {e}")
        raise
    
    yield
    
    logger.info("Cerrando sistema...")
    system_components['active_sessions'].clear()
    logger.info("Sistema cerrado correctamente")

app = FastAPI(
    title="Sistema de Orientacion Vocacional v2.0",
    description="API para orientacion vocacional con ML adaptativo y explicaciones personalizadas",
    version="2.0.0",
    lifespan=lifespan
)

if CONFIG['enable_cors']:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

async def initialize_system_components():
    models_path = CONFIG['models_path']
    
    logger.info("Inicializando ML Predictor...")
    system_components['ml_predictor'] = VocationalMLPredictor(
        modelo_capacidad_path=f"{models_path}/modelo_capacidad.pkl",
        modelo_rama_path=f"{models_path}/modelo_rama.pkl", 
        modelo_opciones_path=f"{models_path}/modelo_opciones.pkl",
        opciones_info_path=f"{models_path}/opciones_vocacionales.json"
    )
    
    logger.info("Inicializando Adaptive Questionnaire...")
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
    
    if CONFIG['enable_llm']:
        logger.info("Inicializando LLM Explainer...")
        system_components['llm_explainer'] = VocationalLLMExplainer(
            api_key=CONFIG['groq_api_key']
        )
    else:
        logger.info("LLM Explainer deshabilitado")
        system_components['llm_explainer'] = VocationalLLMExplainer(api_key=None)

def get_ml_predictor() -> VocationalMLPredictor:
    if system_components['ml_predictor'] is None:
        raise HTTPException(status_code=503, detail="ML Predictor no inicializado")
    return system_components['ml_predictor']

def get_questionnaire() -> AdaptiveQuestionnaire:
    if system_components['questionnaire'] is None:
        raise HTTPException(status_code=503, detail="Questionnaire no inicializado")
    return system_components['questionnaire']

def get_llm_explainer() -> VocationalLLMExplainer:
    if system_components['llm_explainer'] is None:
        raise HTTPException(status_code=503, detail="LLM Explainer no inicializado")
    return system_components['llm_explainer']

async def cleanup_expired_sessions():
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_info in system_components['active_sessions'].items():
                if 'timestamp' in session_info:
                    session_time = datetime.fromisoformat(session_info['timestamp'])
                    if current_time - session_time > timedelta(hours=CONFIG['session_timeout_hours']):
                        expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del system_components['active_sessions'][session_id]
                logger.info(f"Sesion expirada removida: {session_id}")
            
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error en cleanup de sesiones: {e}")
            await asyncio.sleep(3600)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html><html><head><title>Sistema de Orientacion Vocacional v2.0</title><meta charset="UTF-8"><style>body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px}.header{text-align:center;color:#2c3e50}.endpoint{background:#f8f9fa;padding:15px;margin:10px 0;border-radius:5px}.method{display:inline-block;padding:3px 8px;border-radius:3px;color:white;font-weight:bold}.post{background:#28a745}.get{background:#007bff}.stats{background:#e9ecef;padding:15px;border-radius:5px;margin:20px 0}</style></head><body><div class="header"><h1>Sistema de Orientacion Vocacional v2.0</h1><p>API inteligente con Machine Learning</p></div><div class="stats"><h3>Estado del Sistema</h3><p><strong>Estado:</strong> Operativo</p><p><strong>Version:</strong> 2.0.0</p></div><h3>Endpoints Disponibles</h3><div class="endpoint"><span class="method post">POST</span> <code>/api/v2/session/start</code></div><div class="endpoint"><span class="method get">GET</span> <code>/api/v2/session/{session_id}/next-question</code></div><div class="endpoint"><span class="method post">POST</span> <code>/api/v2/session/{session_id}/answer</code></div><div class="endpoint"><span class="method post">POST</span> <code>/api/v2/session/{session_id}/finish</code></div><div class="endpoint"><span class="method get">GET</span> <code>/docs</code></div></body></html>"""

@app.get("/api/v2/health")
async def health_check():
    try:
        ml_status = system_components['ml_predictor'] is not None
        questionnaire_status = system_components['questionnaire'] is not None
        llm_status = system_components['llm_explainer'] is not None
        
        return {
            "status": "healthy" if all([ml_status, questionnaire_status, llm_status]) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "ml_predictor": "ok" if ml_status else "error",
                "questionnaire": "ok" if questionnaire_status else "error", 
                "llm_explainer": "ok" if llm_status else "error"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="System health check failed")

@app.post("/api/v2/session/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    try:
        if len(system_components['active_sessions']) >= CONFIG['max_active_sessions']:
            raise HTTPException(status_code=429, detail="Limite de sesiones alcanzado")
        
        session = questionnaire.iniciar_sesion()
        
        system_components['active_sessions'][session.session_id] = {
            'session': session,
            'timestamp': datetime.now().isoformat(),
            'metadata': request.metadata
        }
        
        logger.info(f"Sesion iniciada: {session.session_id}")
        
        return SessionStartResponse(
            session_id=session.session_id,
            message="Sesion iniciada exitosamente",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error iniciando sesion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/session/{session_id}/next-question", response_model=NextQuestionResponse)
async def get_next_question(
    session_id: str,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    try:
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesion no encontrada")
        
        pregunta_data = questionnaire.obtener_siguiente_pregunta(session_id)
        
        if pregunta_data is None:
            return NextQuestionResponse(
                pregunta=None,
                metadata={"debe_terminar": True},
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
        logger.error(f"Error obteniendo pregunta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/session/{session_id}/answer", response_model=AnswerResponse)
async def submit_answer(
    session_id: str,
    answer: AnswerRequest,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire)
):
    try:
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesion no encontrada")
        
        resultado = questionnaire.enviar_respuesta(
            session_id=session_id,
            pregunta_id=answer.pregunta_id,
            respuesta=answer.respuesta
        )
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/session/{session_id}/finish", response_model=FinishResponse)
async def finish_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire),
    llm_explainer: VocationalLLMExplainer = Depends(get_llm_explainer)
):
    try:
        if session_id not in system_components['active_sessions']:
            raise HTTPException(status_code=404, detail="Sesion no encontrada")
        
        session_info = system_components['active_sessions'][session_id]
        session = session_info['session']
        
        logger.info(f"Finalizando sesion {session_id}")
        
        try:
            resultado_ml = questionnaire.finalizar_cuestionario(session_id)
        except Exception as e:
            logger.error(f"Error ML predictor: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error ML: {str(e)}")
        
        top_n = CONFIG['top_n_recomendaciones']
        resultado_ml['recomendaciones'] = resultado_ml['recomendaciones'][:top_n]
        
        reporte_completo = {}
        try:
            if llm_explainer.cliente_activo:
                logger.info(f"Generando LLM para {len(resultado_ml['recomendaciones'])} recomendaciones")
                reporte_completo = llm_explainer.generar_reporte_completo(
                    recomendaciones=resultado_ml['recomendaciones'],
                    metadata_estudiante={
                        'capacidad_academica': resultado_ml.get('capacidad_academica', {}),
                        'categoria': resultado_ml.get('capacidad_academica', {}).get('categoria', 'N/A'),
                        'rama_universitaria': resultado_ml.get('rama_universitaria')
                    }
                )
            else:
                logger.warning("LLM no disponible")
                reporte_completo = {
                    'resumen_ejecutivo': 'Reporte generado sin LLM',
                    'mensaje_motivacional': 'Explora tus opciones vocacionales',
                    'recomendaciones': []
                }
        except Exception as e:
            logger.error(f"Error LLM: {e}")
            logger.error(traceback.format_exc())
            reporte_completo = {
                'resumen_ejecutivo': 'Reporte sin explicaciones LLM',
                'mensaje_motivacional': 'Continua explorando',
                'recomendaciones': []
            }
        
        top_3_recomendaciones = []
        for i, rec in enumerate(resultado_ml['recomendaciones']):
            explicacion_llm = ""
            explicacion_generada = False
            
            if i < len(reporte_completo.get('recomendaciones', [])):
                rec_llm = reporte_completo['recomendaciones'][i]
                explicacion_llm = rec_llm.get('explicacion_personalizada', '')
                explicacion_generada = rec_llm.get('explicacion_generada', False)
            
            info_basica = rec.get('info_basica', {})
            
            recomendacion_formateada = {
                "ranking": rec.get('ranking', i + 1),
                "nombre": rec.get('nombre', 'Desconocido'),
                "codigo": rec.get('codigo', ''),
                "match_score": rec.get('match_score', 0.0),
                "match_porcentaje": rec.get('match_score_porcentaje', '0%'),
                "categoria": rec.get('categoria', ''),
                "rama": rec.get('rama'),
                "info_basica": {
                    "anos_estudio": info_basica.get('años_estudio') or info_basica.get('anos_estudio'),
                    "nivel_educativo": info_basica.get('nivel_educativo', ''),
                    "dificultad": info_basica.get('dificultad', ''),
                    "salario_promedio": info_basica.get('salario_promedio')
                },
                "explicacion_llm": explicacion_llm,
                "explicacion_generada_por_ia": explicacion_generada
            }
            top_3_recomendaciones.append(recomendacion_formateada)
        
        capacidad = resultado_ml.get('capacidad_academica', {})
        
        response = FinishResponse(
            session_id=session_id,
            capacidad_academica={
                "score": capacidad.get('score', 0.0),
                "categoria": capacidad.get('categoria', 'N/A'),
                "descripcion": capacidad.get('descripcion', '')
            },
            rama_universitaria=resultado_ml.get('rama_universitaria'),
            top_3_recomendaciones=top_3_recomendaciones,
            resumen_ejecutivo=reporte_completo.get('resumen_ejecutivo', ''),
            mensaje_motivacional=reporte_completo.get('mensaje_motivacional', ''),
            total_respuestas=len(session.respuestas),
            timestamp=datetime.now().isoformat(),
            llm_disponible=llm_explainer.cliente_activo
        )
        
        background_tasks.add_task(cleanup_session, session_id)
        logger.info(f"Sesion finalizada: {session_id}")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finish: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/v2/session/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in system_components['active_sessions']:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    
    session_info = system_components['active_sessions'][session_id]
    session = session_info['session']
    
    return {
        'session_id': session_id,
        'status': 'active',
        'total_respuestas': len(session.respuestas),
        'fase_actual': session.fase_actual
    }

@app.get("/api/v2/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    ml_predictor: VocationalMLPredictor = Depends(get_ml_predictor),
    questionnaire: AdaptiveQuestionnaire = Depends(get_questionnaire),
    llm_explainer: VocationalLLMExplainer = Depends(get_llm_explainer)
):
    return SystemStatsResponse(
        ml_system=ml_predictor.get_estadisticas_sistema(),
        questionnaire_system=questionnaire.obtener_estadisticas_sistema(),
        llm_system=llm_explainer.get_estadisticas_uso(),
        active_sessions=len(system_components['active_sessions'])
    )

async def cleanup_session(session_id: str):
    try:
        if session_id in system_components['active_sessions']:
            del system_components['active_sessions'][session_id]
            logger.info(f"Sesion limpiada: {session_id}")
    except Exception as e:
        logger.error(f"Error cleanup: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,           # ← UN SOLO WORKER
        reload=False,        # ← SIN RELOAD EN PRODUCCIÓN
        log_level="info",
        timeout_keep_alive=180
    )