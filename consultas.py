import requests
import time
import pyfiglet
import os

# URL a la cual quieres hacer la solicitud GET
base_url = 'http://localhost:3000/api/sim/execute/'
big_font = pyfiglet.Figlet(font='big')

for iteracion in range(0,10):
    # modo de recompensa
    for modo in range(3, 4):
        for config in range(3,4):
            url = base_url + str(modo) + "/" + str(config)

            respuesta = requests.get(url)

            # Verificar si la solicitud fue exitosa
            if respuesta.status_code == 200:
                # Procesar la respuesta, por ejemplo, como JSON
                datos = respuesta.json()
                print(url)
                os.system('cls' if os.name == 'nt' else 'clear')
                big_text = big_font.renderText(str(iteracion))

                print(big_text)
            else:
                print("Error en la solicitud: Código de estado", respuesta.status_code)

            ocupado = True
            while ocupado:
                respuesta_ocupado = requests.get('http://localhost:3000/api/sim/ocupado')
                ocupado_aux = respuesta_ocupado.json()

                if ocupado_aux:
                    time.sleep(5)
                else:
                    ocupado = False