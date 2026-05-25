import cv2 # OpenCV, responsável pelo processamento de imagem e comunicação com a câmera
import time # Biblioteca nativa do Python para manipulação de tempo (cronômetro e cálculo de FPS)

# ==========================================
# CONFIGURAÇÕES PRINCIPAIS
# ==========================================

# Define qual porta USB de vídeo o sistema deve usar (1 geralmente é a câmera externa, 0 a webcam embutida do notebook)
indice_camera = 1 

# Define a duração máxima do loop de gravação
tempo_limite_segundos = 15

# O comando cv2.VideoCapture abre o fluxo de dados da câmera escolhida
# O argumento cv2.CAP_DSHOW força o uso da API "DirectShow" no Windows, garantindo acesso estável às configurações nativas da câmera
cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

# 1. FORÇAR O FORMATO MJPG (formato de compressão)
# Cria um identificador interno de 4 bytes (FourCC) que representa o codec MJPG (Motion JPEG)
fourcc_cam = cv2.VideoWriter_fourcc(*'MJPG')
# Envia a ordem para a câmera compactar as imagens na própria placa dela antes de enviar pelo cabo USB, evitando gargalo na transferência
# As funções cap.set enviam as variáveis definidas acima diretamente para os registradores de hardware da câmera.
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cam)

# 2. Configurar Resolução (largura e altura do vídeo) e FPS desejado
largura = 640 
altura = 480 
fps_desejado = 30

# As funções cap.set enviam as variáveis definidas acima diretamente para os registradores de hardware da câmera.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura) # Aplica a largura.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura) # Aplica a altura.
cap.set(cv2.CAP_PROP_FPS, fps_desejado) # Aplica a restrição de FPS.

# 3. TRAVAR OS PARÂMETROS AUTOMÁTICOS 
# FIXO - Desativa a exposição automática para que a câmera não tente corrigir o brilho (A depender da câmera pode ser 0, e não 0.25).
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
# MUTÁVEL - Fixa a abertura/tempo do obturador. Valores negativos escurecem a imagem mas evitam quedas de FPS.
cap.set(cv2.CAP_PROP_EXPOSURE, -5) 
# FIXO - Desativa o balanço de branco automático para impedir que a câmera altere artificialmente a proporção RGB lida.
cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
# MUTÁVEL - Fixa a leitura de cor da lente em 4000 Kelvin (uma luz neutra/quente contínua).
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000) 
# FIXA - Desativa o motor da lente de foco automático para que ela não fique procurando foco no escuro.
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
# MUTÁVEL - Trava o motor de foco em um ponto específico (0 geralmente foca no infinito).
cap.set(cv2.CAP_PROP_FOCUS, 0) 

# 4. CONFIGURAR A GRAVAÇÃO DO VÍDEO NO PC
nome_arquivo = 'teste_amostra_com_tempo.avi' # String que define o nome exato do arquivo que será criado na pasta do script.
# Define o codec de compressão do arquivo que será salvo no seu HD (XVID é comum para .avi).
fourcc_out = cv2.VideoWriter_fourcc(*'XVID') 
# Cria o objeto gravador de vídeo (out), dizendo a ele qual o nome do arquivo, codec, FPS e dimensões ele deve esperar receber.
out = cv2.VideoWriter(nome_arquivo, fourcc_out, fps_desejado, (largura, altura)) 

# Imprime um aviso no terminal informando que o programa está prestes a entrar no loop de gravação.
print(f"Iniciando gravação por {tempo_limite_segundos} segundos...")

# Grava na variável tempo_inicio a hora exata (em segundos corridos) que o relógio interno do PC marcava antes do loop começar.
tempo_inicio = time.time() 
# Inicializa em zero a variável que usaremos para armazenar o tempo do quadro passado (vital para calcular o FPS real).
prev_frame_time = 0 

# Inicia um loop infinito. O código gira aqui dentro até ser quebrado.
while True:
    # O comando read() pede a foto mais atual à câmera. 
    # 'ret' vira True se conseguiu ler e 'frame' recebe a matriz de pixels da imagem.
    ret, frame = cap.read() 
    
    # Se 'ret' for falso (a câmera travou ou desconectou), ele quebra (break) o loop para o programa não crashar.
    if not ret:
        print("Erro ao ler o frame da câmera.")
        break
    
    # Subtrai o momento atual pelo momento inicial para descobrir quantos segundos exatos se passaram desde o início do loop.
    tempo_decorrido = time.time() - tempo_inicio
    
    # Se o tempo que passou atingiu ou ultrapassou o limite definido pelo usuário, quebra o loop para parar de gravar.
    if tempo_decorrido >= tempo_limite_segundos:
        print("\nTempo limite atingido. Finalizando gravação...")
        break
    
    # Pega o 'frame' limpo (sem nenhum texto desenhado) e escreve (write) dentro do arquivo .avi no seu disco rígido.
    out.write(frame)
    
    # Marca exatamente o momento em milissegundos em que esta imagem atual foi processada para calcular a taxa na tela.
    new_frame_time = time.time()
    
    # Bloco try/except (Tente fazer / Se der erro faça outra coisa) para prevenir o programa de fechar.
    try:
        # A fórmula do FPS é 1 segundo dividido pela diferença de tempo entre o quadro passado e este novo quadro.
        fps_real = 1 / (new_frame_time - prev_frame_time)
    except ZeroDivisionError:
        # Se os quadros vierem tão rápido que a diferença de tempo pareça zero, o Python avisa erro matemático.
        # Capturamos esse erro aqui e temporariamente definimos o FPS para 0 para não fechar o programa.
        fps_real = 0
        
    # Salva o tempo do quadro atual na variável de tempo anterior, para a próxima volta do loop usá-la na conta matemática.
    prev_frame_time = new_frame_time
    
    # Calcula quantos segundos faltam para encerrar. A função max(0, ...) garante que o número nunca mostre um valor negativo.
    tempo_restante = max(0, tempo_limite_segundos - tempo_decorrido)
    
    # Desenha na imagem (frame) o FPS inteiro (sem casas decimais), na coordenada 10x40, verde e espessura 2.
    cv2.putText(frame, f"FPS: {int(fps_real)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Desenha na imagem (frame) o tempo restante com 1 casa decimal, na coordenada 10x80, vermelho.
    cv2.putText(frame, f"Tempo: {tempo_restante:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Pega o frame (agora sujo com os textos em vermelho e verde) e mostra em uma janela no monitor do computador.
    cv2.imshow('Camera Logitech - Gravacao Temporizada', frame)
    
    # Pausa a execução por 1 milissegundo. Se durante esse 1ms a letra 'q' (quit) for pressionada, quebra o loop manualmente.
    # O & 0xFF limpa a entrada do teclado para garantir que o sistema entenda perfeitamente a tecla apertada.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nGravação interrompida manualmente.")
        break

# --- O PROGRAMA SÓ CHEGA AQUI QUANDO O LOOP ACIMA É QUEBRADO (BREAK) ---

# Libera o uso da porta USB da câmera, permitindo que outros programas possam usá-la em seguida.
cap.release()
# Fecha o arquivo .avi gravado no disco, montando a estrutura final de metadados do vídeo para que ele possa ser lido.
out.release()
# Destrói e fecha todas as janelas que o comando cv2.imshow abriu na sua área de trabalho.
cv2.destroyAllWindows()

# Imprime o aviso final no terminal de que tudo rodou bem e o arquivo está salvo e pronto.
print(f"Vídeo salvo com sucesso como '{nome_arquivo}'")
