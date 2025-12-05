import sys
import os
import subprocess
import argparse
import asyncio
import aiohttp
import json
from pathlib import Path
import time

def load_env_file():
    env_file = Path('.env')
    
    if not env_file.exists():
        print("Archivo .env no encontrado")
        return False
    
    print("Cargando configuración desde .env...")
    
    with open(env_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Corregir paths que empiecen con /outputs
                if value.startswith('/outputs'):
                    value = value[1:]  # Quitar la barra inicial
                
                os.environ[key] = value
                display_value = '********' if 'KEY' in key or 'PASSWORD' in key else value
                print(f"   {key} = {display_value}")
    
    print("Archivo .env cargado correctamente")
    return True

def check_dependencies():
    print("Verificando dependencias...")
    
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'scikit-learn': 'sklearn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'pydantic': 'pydantic',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   OK {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"   FALTA {package_name}")
    
    if missing_packages:
        print(f"Faltan dependencias: {', '.join(missing_packages)}")
        print("Instala con: pip install -r requirements.txt")
        return False
    
    print("Todas las dependencias están instaladas")
    return True

def check_model_files():
    print("Verificando archivos de modelos...")
    
    models_path = os.getenv('MODELS_PATH', 'outputs')
    base_path = Path(models_path)
    
    required_files = [
        "modelo_capacidad.pkl",
        "modelo_rama.pkl", 
        "modelo_opciones.pkl",
        "opciones_vocacionales.json",
        "preguntas-fase1.json",
        "rama1_salud_preguntas_fase2.json",
        "rama2_ingenieria_preguntas_fase2.json",
        "rama3_negocios_preguntas_fase2.json",
        "rama4_oficios_tecnicos_preguntas_fase2.json",
        "rama5_oficios_basicos_preguntas_fase2.json"
    ]
    
    missing_files = []
    
    for filename in required_files:
        filepath = base_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   OK {filename} ({size_mb:.1f} MB)")
        else:
            missing_files.append(filename)
            print(f"   FALTA {filename}")
    
    if missing_files:
        print(f"Faltan archivos: {', '.join(missing_files)}")
        return False
    
    print("Todos los archivos de modelo están disponibles")
    return True

def check_configuration():
    print("Verificando configuración...")
    
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and len(groq_key.strip()) > 10:
        print("   OK GROQ_API_KEY configurada")
        print(f"       Key: {groq_key[:8]}...{groq_key[-4:]} (parcial)")
    else:
        print("   ADVERTENCIA GROQ_API_KEY no configurada (usará explicaciones básicas)")
        print("       Obtén una key gratis en: https://console.groq.com/keys")
    
    models_path = os.getenv('MODELS_PATH', 'outputs')
    if os.path.exists(models_path):
        print(f"   OK MODELS_PATH: {models_path}")
    else:
        print(f"   ERROR MODELS_PATH no existe: {models_path}")
        return False
    
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"   OK HOST: {host}")
    print(f"   OK PORT: {port}")
    print(f"   OK DEBUG: {debug}")
    
    enable_cors = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
    cors_origins = os.getenv('CORS_ORIGINS', '*')
    
    print(f"   OK CORS habilitado: {enable_cors}")
    print(f"   OK CORS origins: {cors_origins}")
    
    print("Configuración verificada")
    return True

async def test_api_endpoints():
    print("Probando endpoints de la API...")
    
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    base_url = f"http://localhost:{port}"
    
    async with aiohttp.ClientSession() as session:
        try:
            print("   Test 1: Health Check...")
            async with session.get(f"{base_url}/api/v2/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"      OK Status: {data.get('status', 'unknown')}")
                else:
                    print(f"      ERROR Health check failed: {response.status}")
                    return False
            
            print("   Test 2: System Stats...")
            async with session.get(f"{base_url}/api/v2/system/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"      OK Sesiones activas: {data.get('active_sessions', 0)}")
                else:
                    print(f"      ERROR System stats failed: {response.status}")
                    return False
            
            print("   Test 3: Start Session...")
            session_payload = {"metadata": {"test": True}}
            async with session.post(f"{base_url}/api/v2/session/start", 
                                   json=session_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    session_id = data.get('session_id')
                    print(f"      OK Sesión creada: {session_id}")
                else:
                    print(f"      ERROR Start session failed: {response.status}")
                    return False
            
            print("   Test 4: Get Next Question...")
            async with session.get(f"{base_url}/api/v2/session/{session_id}/next-question") as response:
                if response.status == 200:
                    data = await response.json()
                    pregunta = data.get('pregunta')
                    if pregunta:
                        print(f"      OK Pregunta obtenida: {pregunta.get('id', 'N/A')}")
                        pregunta_id = pregunta.get('id')
                    else:
                        print(f"      ADVERTENCIA No hay preguntas disponibles")
                        return False
                else:
                    print(f"      ERROR Get question failed: {response.status}")
                    return False
            
            print("   Test 5: Submit Answer...")
            answer_payload = {"pregunta_id": pregunta_id, "respuesta": "A"}
            async with session.post(f"{base_url}/api/v2/session/{session_id}/answer",
                                   json=answer_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"      OK Respuesta registrada: {data.get('total_respuestas', 0)} total")
                else:
                    print(f"      ERROR Submit answer failed: {response.status}")
                    return False
            
            print("   Test 6: Demo Prediction...")
            async with session.get(f"{base_url}/api/v2/demo") as response:
                if response.status == 200:
                    data = await response.json()
                    top_rec = data.get('resultado', {}).get('recomendaciones', [])
                    if top_rec:
                        print(f"      OK Demo completado: {top_rec[0].get('nombre', 'N/A')} como top")
                    else:
                        print(f"      ADVERTENCIA Demo sin recomendaciones")
                else:
                    print(f"      ERROR Demo failed: {response.status}")
                    return False
            
            print("Todos los tests de API pasaron exitosamente")
            return True
            
        except aiohttp.ClientError as e:
            print(f"Error de conexión: {e}")
            print("   Asegúrate de que el servidor esté ejecutándose")
            return False
        except Exception as e:
            print(f"Error inesperado en tests: {e}")
            return False

def wait_for_server(max_wait=30):
    port = os.getenv('PORT', '8000')
    url = f"http://localhost:{port}/api/v2/health"
    
    print(f"Esperando servidor en {url} (máximo {max_wait}s)...")
    
    import requests
    
    for i in range(max_wait):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("Servidor disponible")
                return True
        except:
            pass
        
        if i % 5 == 0 and i > 0:
            print(f"   Esperando... ({i}s)")
        
        time.sleep(1)
    
    print("Timeout esperando servidor")
    return False

def start_server():
    print("Iniciando servidor API...")
    
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    try:
        cmd = [
            "python", "-m", "uvicorn", 
            "api_server:app",
            "--host", host,
            "--port", port
        ]
        
        if debug:
            cmd.append("--reload")
        
        print(f"Comando: {' '.join(cmd)}")
        print(f"Servidor disponible en: http://localhost:{port}")
        print(f"Documentación en: http://localhost:{port}/docs")
        print(f"Health check: http://localhost:{port}/api/v2/health")
        print("Presiona Ctrl+C para detener el servidor")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("Servidor detenido por el usuario")
    except Exception as e:
        print(f"Error iniciando servidor: {e}")

def start_server_production():
    print("Iniciando servidor en modo PRODUCCIÓN...")
    
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    workers = os.getenv('WORKERS', '4')
    timeout = os.getenv('TIMEOUT', '30')
    
    try:
        cmd = [
            "python", "-m", "gunicorn",
            "api_server:app",
            "-w", workers,
            "-k", "uvicorn.workers.UvicornWorker",
            "--bind", f"{host}:{port}",
            "--timeout", timeout
        ]
        
        print(f"Comando: {' '.join(cmd)}")
        print(f"Servidor de producción en: http://localhost:{port}")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("Servidor detenido por el usuario")
    except Exception as e:
        print(f"Error iniciando servidor de producción: {e}")

async def run_full_test():
    print("EJECUTANDO SUITE COMPLETA DE PRUEBAS")
    print("=" * 50)
    
    host = os.getenv('HOST', '127.0.0.1')
    port = os.getenv('PORT', '8000')
    
    server_process = None
    
    try:
        print("Iniciando servidor para pruebas...")
        server_process = subprocess.Popen([
            "python", "-m", "uvicorn",
            "api_server:app",
            "--host", host,
            "--port", port
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if wait_for_server():
            success = await test_api_endpoints()
            
            if success:
                print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
                print("El sistema está listo para usar")
            else:
                print("ALGUNAS PRUEBAS FALLARON")
                print("Revisa la configuración y logs")
                
            return success
        else:
            print("No se pudo conectar al servidor para pruebas")
            return False
            
    except Exception as e:
        print(f"Error en suite de pruebas: {e}")
        return False
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()

def main():
    parser = argparse.ArgumentParser(description="Sistema de Orientación Vocacional v2.0")
    parser.add_argument('--test', action='store_true', 
                       help='Ejecutar pruebas completas')
    parser.add_argument('--check', action='store_true',
                       help='Solo verificar configuración')
    parser.add_argument('--production', action='store_true',
                       help='Iniciar en modo producción')
    
    args = parser.parse_args()
    
    print("SISTEMA DE ORIENTACIÓN VOCACIONAL v2.0")
    print("=" * 50)
    
    load_env_file()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_model_files():
        sys.exit(1)
    
    if not check_configuration():
        sys.exit(1)
    
    if args.check:
        print("Todas las verificaciones pasaron")
        print("Sistema listo para ejecutar")
        return
    
    if args.test:
        print("Ejecutando suite de pruebas...")
        success = asyncio.run(run_full_test())
        sys.exit(0 if success else 1)
    
    if args.production:
        start_server_production()
    else:
        start_server()

if __name__ == "__main__":
    main()