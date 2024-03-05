import logging
import gradio as gr 
from ctransformers import AutoModelForCausalLM
import threading
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
        time.sleep(2)  # Esperar 2 segundos antes de la próxima impresión

# Crear y empezar un hilo para el monitoreo de recursos
monitoring_thread = threading.Thread(target=print_resource_usage)
monitoring_thread.daemon = True  # Establecer como un daemon para que no bloquee el programa de terminar
monitoring_thread.start()


# Configuración del logger
logging.basicConfig(filename='coderIA_logs.txt', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

system_message = """
[INST]
Dado el contexto proporcionado y la última consulta del usuario, genera una solución de código concisa y precisa que aborde el desafío de programación específico. Asegúrate de que la respuesta incluya:
1. Una breve explicación de tu enfoque.
2. La solución como un fragmento de código completo, encerrado en triple comillas inversas.
3. Reflexiones sobre posibles mejoras o estrategias alternativas.
[/INST]
"""

# Cargar el modelo una sola vez
model_instance = None

def load_llm():
    
    global model_instance
    #model_path = "./models/llama-2-7b-chat.Q5_K_M.gguf"
    model_path = "./models/codellama-7b-instruct.Q5_K_M.gguf"

    if model_instance is None:
        print("Cargando el modelo...")
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type='llama',
            max_new_tokens=2048,  # Máximo de nuevos tokens a generar.
            temperature=0.1,  # La temperatura a usar para el muestreo.
            repetition_penalty=1.13,  # El penalty de repetición para el muestreo.
            context_length=4096,  # La longitud máxima de contexto a usar.
            gpu_layers=110  # Número de capas a ejecutar en GPU.
        )
        print("Modelo cargado éxitosamente")
    else:
        print("Ya se ha cargado el modelo anteriormente")
    
    return model_instance
def truncate_history(history, max_tokens=1024):
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
    new_history += "\nAnswer:"
    return new_history
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
        
        chat_history = truncate_history(chat_history + [(message, "")])  # Añadir el mensaje actual al historial
        
        print("Historial del chat antes de formateo")
        for i in chat_history:
            print (i)
        print("Fin de historial previo a formateo")
        formatted_history = format_history_for_model(chat_history)  # Formatear el historial para el modelo
        
        full_message = f"{formatted_history}\n{system_message}"
        print (f"Mensaje completo: {full_message}")
        response = llm(full_message)  # Generar respuesta
        cleaned_response = clean_response(response)
        
        logging.info(f"Message: {message}")
        logging.info(f"Response: {response}")
        return cleaned_response
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

      
title = "CoderIA"

description = """
CoderIA es un modelo de lenguaje que funciona como un asistente de c&oacute;digo, con el que podr&aacute;s programar en la mayor&iacute;a de lenguajes de programaci&oacute;n.Si desconoces el uso, desde este modelo, aqu&iacute; tienes algunas instrucciones:
<ul>
<li>Para que el c&oacute;digo funcione completamente , en ocasiones puede necesitar varias revisiones. Para ello realiza las preguntas poco a poco hasta conseguir el resultado que necesites.</li>
<li>Usa este modelo bajo tu responsabilidad.</li>
<li>Si no obtienes respuesta a tu consulta, intenta reformularla.</li>
<li>Si necesitas alguna aclaraci&oacute;n o tienes cualquier otra duda, puedes enviar un correo electr&oacute;nico a soporte: <a href='mailto:vjimmes2003@g.educaand.es'> Mandar email </a>. o revisar nuestras gu&iacute;as: <a href='http://fuentezuelas.com/ia/playground/ChatterIA/index.php#guias' target="_blank"> Acceder a gu&iacute;as </a></li>
<li>Es importante que sepas este sitio es de uso educativo. El env&iacute;o de mensajes masivos puede dar error.</li>
</ul>
"""

theme = 'Taithrah/Minimal'

examples = [
    'Escribe un programa en Python que solicite al usuario ingresar su nombre y edad, luego imprime un mensaje que diga "Hola [nombre], tienes [edad] años".',
    'Desarrolla una función en JavaScript que tome un arreglo de números y devuelva otro arreglo solo con los números impares.',
    'Crea una clase en Java llamada "CuentaBancaria" que tenga como atributos "titular" y "saldo". Agrega métodos para depositar y retirar dinero.',
    'Escribe un script en Python que lea un archivo de texto grande, cuente la frecuencia de cada palabra y almacene los resultados en un diccionario.',
    'Desarrolla una aplicación sencilla en Flask que tenga una ruta de inicio y otra ruta que acepte parámetros URL para mostrar mensajes personalizados al usuario.'
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
    chatbot=gr.Chatbot(label="CoderIA", show_share_button=True),
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
).launch(server_name='0.0.0.0', server_port=7880)