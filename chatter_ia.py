import logging
import gradio as gr 
from ctransformers import AutoModelForCausalLM
import threading
from langdetect import detect, DetectorFactory
from langid import classify
import spacy
import os
import psutil
import time

# Obtener el proceso actual
process = psutil.Process(os.getpid())

# Función para imprimir el uso de CPU y RAM
def print_resource_usage():
    while True:  # Crear un bucle infinito para que se ejecute continuamente
        cpu_usage = process.cpu_percent(interval=1)
        ram_usage = process.memory_info().rss / (1024 * 1024)  # Convertir bytes a MB
        pid = str(process.pid)
        print(f"Uso de CPU: {cpu_usage}%")
        print(f"Uso de RAM: {ram_usage} MB")
        print(f"PID: {pid}")
        time.sleep(10)  # Esperar 2 segundos antes de la próxima impresión

# Crear y empezar un hilo para el monitoreo de recursos
monitoring_thread = threading.Thread(target=print_resource_usage)
monitoring_thread.daemon = True  # Establecer como un daemon para que no bloquee el programa de terminar
monitoring_thread.start()


# Configuración del logger
logging.basicConfig(filename='chatterIA_logs.txt', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

DetectorFactory.seed = 0  # Para obtener resultados consistentes en la detección del idioma (outputs)

# Introducimos un mensaje para el modelo para que hable solo en español y con respeto
system_message ="Eres un asistente que siempre responde en español. A cualquiera de sus peticiones y nunca hablas en ningún otro idioma"

# Cargar el modelo una sola vez
model_instance = None

# Cargar el modelo que detecta el idioma de los mensajes solo 1 vez. (inputs)
nlp_sm = spacy.load("es_core_news_sm")
def load_llm():
    
    global model_instance
    model_path = "./models/wizardlm-13b-v1.2.Q3_K_M.gguf"

    if model_instance is None:
        print("Cargando el modelo...")
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type='llama',
            max_new_tokens=2048,  # Máximo de nuevos tokens a generar.
            temperature=0.1,  # La temperatura a usar para el muestreo.
            repetition_penalty=1.13,  # El penalty de repetición para el muestreo.
            context_length=4096,  # La longitud máxima de contexto a usar.
            gpu_layers=130  # Número de capas a ejecutar en GPU.
        )
        print("Modelo cargado éxitosamente")
    else:
        print("Ya se ha cargado el modelo anteriormente")
    
    return model_instance
def truncate_history(history, max_tokens=2048):
    truncated_history = []
    total_tokens = 0
    for user_msg, bot_msg in reversed(history):
        # Asegurarse de que ni user_msg ni bot_msg sean None
        user_msg = user_msg if user_msg is not None else ""
        bot_msg = bot_msg if bot_msg is not None else ""
        
        msg_tokens = len(user_msg.split()) + len(bot_msg.split())
        if total_tokens + msg_tokens > max_tokens:
            break
        truncated_history.insert(0, (user_msg, bot_msg))
        total_tokens += msg_tokens
    return truncated_history
def clean_response(response):
    code_block_count = response.count("```")
    
    if code_block_count % 2 != 0:
        last_occurrence_index = response.rfind("```")
        response = response[:last_occurrence_index] + response[last_occurrence_index+3:]
    
    return response

def format_history_for_model(history):
    new_history = "\n".join(f"{user_msg}\n{bot_msg}".strip() for user_msg, bot_msg in history)
    new_history += "\nRespuesta:"
    return new_history


def detectar_idioma_con_spacy(texto):
    doc = nlp_sm(texto, disable=["parser","ner"])
    return True

def prepare_message(message):
    # Primero, intenta detectar el idioma con langid
    lang, confidence = classify(message)
    if lang == 'es':
        return str(message)  # Si langid está seguro de que es español, retorna el mensaje
    
    # Si el mensaje es muy corto y langid no lo clasifica definitivamente como español,
    # utiliza SpaCy como una verificación adicional.
    if len(message.split()) <= 3:  # Considera ajustar este umbral según tus necesidades
        es_espanol_spacy = detectar_idioma_con_spacy(message)
        if es_espanol_spacy:
            return str(message)  # Si SpaCy procesa el texto sin problemas, asume que es español
    
    # Si ninguna de las verificaciones anteriores concluye que el mensaje es en español, retorna None
    return None

def llm_function(message, chat_history ):
    try:
        print(f"Mensaje principal:{message}")
        logging.info(f"Tipo de Message: {type(message)}, Message: {message}")
        message = str(message)
        message_str = str(message)

        # Aquí puedes añadir cualquier validación adicional del mensaje
        if not message_str.strip():
            logging.info("El mensaje está vacío o solo contiene espacios.")
            return "Por favor, envía un mensaje no vacío.", chat_history

        # Continuar con el procesamiento...
        logging.info(f"Message: {message_str}")
        
        print(f"Historial del Chat (segun el chatbot)")
        for i in chat_history:
            print (i)
        print("Fin de variables desde el chat")
            
        llm = load_llm()
        message = prepare_message(message)
        
        if message is None:
            return f"Por favor, envía tu mensaje en español. Actualmente idioma es: {detect(message)}"
        chat_history = truncate_history(chat_history + [(message, "")])  # Añadir el mensaje actual al historial
        
        print("Historial del chat antes de formateo")
        for i in chat_history:
            print (i)
        print("Fin de historial previo a formateo")
        formatted_history = format_history_for_model(chat_history)  # Formatear el historial para el modelo
        
        full_message = f"{system_message}\n{formatted_history}"
        print (f"Mensaje completo: {full_message}")
        response = llm(full_message)  # Generar respuesta
        cleaned_response = clean_response(response)
        
        logging.info(f"Message: {message}")
        logging.info(f"Response: {response}")
        
        try:
            response_language = detect(response)
            print(f"Idioma en response_language: {response_language}")
            print(f"Response:{response}")
            if response_language != 'es':
                return "Lo siento, pero no puedo responder correctamente en español a esa pregunta."
            else:
                cleaned_response = clean_response(response)
                return cleaned_response
        except Exception as e:
            logging.error(f"Error al detectar el idioma: {e}")
            return "Hubo un error al procesar tu mensaje."
    except Exception as e:
        logging.error(f"Error al procesar el mensaje:  {e}")
        return "Hubo un error al procesar tu mensaje, revisa que sea un mensaje completo."

def llm_function_with_timeout(message, chat_history, timeout=120):
    result = [None]  # Lista para almacenar el resultado
    def target():
        # Intentar adquirir el semáforo antes de ejecutar la función
        result[0] = llm_function(message, chat_history)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # Si el thread todavía está vivo después del timeout, se asume que está bloqueado y se debe manejar
        thread.join()
        return "Tu pregunta no ha podido ser respondida en el tiempo estimado. Por favor, intenta de nuevo o reformula tu pregunta."
    return result[0]

      
title = "ChatterIA"

description = """
ChatterIA es un modelo de lenguaje que funciona como un asistente con infinidad de conocimientos, pero no infalible. Si desconoces el uso de este modelo, aqu&iacute; tienes unas instrucciones para optimizar su uso:
<ul>
<li>Las respuestas de este modelo pueden no ser siempre precisas, al depender de su configuración, capacidad de interpretación y contextualización</li>
<li>Usa este modelo bajo tu responsabilidad.</li>
<li>Si no o obtienes respuesta a tu consulta, intenta volver a reformularla.</li>
<li>Si necesitas alguna aclaraci&oacute;n o tienes cualquier otra duda, puedes enviar un correo electr&oacute;nico a soporte: <a href='mailto:vjimmes2003@g.educaand.es'> Mandar email </a>. o revisar nuestras gu&iacute;as: <a href='http://fuentezuelas.com/ia/playground/ChatterIA/index.php#guias' target="_blank"> Acceder a gu&iacute;as </a></li>
<li>Es importante que sepas este sitio es de uso educativo. El env&iacute;o de mensajes masivos puede dar error.</li>
</ul>
"""

theme = 'Taithrah/Minimal'

examples = [
    'Escribe un código de python para conectarte con una base de datos en MySQL y liste todas las tablas.',
     'Escribe un poema sobre el otoño con rima consonante y en forma de soneto.',
     'Resume el argumento de "El Quijote", y explícaselo a un niño de 5 años para que le interese.',
     'Dime como fusionar dos diccionarios en Python y diferentes usos que puede tener esto.',
     'Dime que debo considerar al comprar un coche usado y que me recomiendas saber al respecto.',
     'Explica la teoría de la relatividad de Einstein para estudiantes de primaria.',
     'Dime cuáles podrían ser las implicaciones de la inteligencia artificial en el empleo.',
     'Redacta un correo para mi jefe en el cual, quiero pedirle un aumento de salario.',
     'Escribe una cancion que dure aproximadamente 2 minutos y que trate sobre la desigualdad en España.'
]
css = """
#component-0 {
    height:auto !important; 
    min-height:400px !important; 
    overflow:auto !important;
} 
#component-0 .chatbot{
    text-align:justify !important;
    padding: 5px 20px !important; 
}
#component-3 div{
    display:flex;
    justify-content:center;
}
#component-3 h1{
    font-size:2rem;
    text-align:center !important;
}
#component-4 p{ 
    font-size:18px;
    margin-bottom:25px;
}
#component-4 ul{
    padding-left:5%;
}
#component-4 li{
    font-size:16px;
} 
#component-4 li::first-letter{
    margin-left:-10px;
}
.gallery-item{ 
    flex:1 0 33%; 
    align-self:center; 
    height:75px;
}
footer{
    display:none !important;
}
@media only screen and (max-width:600px){
    #component-0 .chatbot{
    text-align:justify !important;
    padding: 5px 20px !important; 
}
    #component-3 h1{
        font-size:1.5rem;
        text-align:center !important;
    }
    #component-4 p{ 
        font-size:14px;
        text-align:justify;
    } 
    #component-4 li{
        font-size:12px;
        paddding:20px;
    }
    #component-4 li::first-letter{
        margin-left:-10px;
    }
    #component-5{
        display:flex;
        flex-direction:row;
    }
    #component-5 button{
        flex: 1 1 30%;
    }
    .gallery-item{ 
        flex:1 0 100%;
        height:100px;
    }
    #component-0 pre, #component-0 code{ 
        max-width: 100%; /* Asegura que el bloque de código no sea más ancho que su contenedor */
        box-sizing: border-box; /* Incluye padding y border dentro del ancho y alto del elemento */
        white-space: pre-wrap; /* Mantiene el formato pero permite el ajuste de palabras */
        word-break: break-all; /* Permite que las palabras se rompan para evitar desbordamiento */
        overflow-x: auto; /* Permite desplazamiento horizontal dentro del bloque de código si es necesario */
        overflow-y: hidden; /* Oculta el desbordamiento vertical, no debería ser necesario con pre-wrap */
    } 
}
"""
demo= gr.ChatInterface(
    fn = llm_function_with_timeout,
    chatbot=gr.Chatbot(label="ChatterIA", show_share_button=True),
    textbox= gr.Textbox(placeholder="Envía tu mensaje...", lines=3, max_lines=10),
    title=title,
    description = description,
    theme = theme,
    examples = examples,
    css = css,
    cache_examples=True,
    submit_btn = "📤 Enviar",
    retry_btn = "🔁 Reenviar",
    undo_btn = "🔙 Deshacer",
    clear_btn= "🗑 Limpiar",
).launch(server_name='0.0.0.0', server_port=7890)