import cv2 # Importa a biblioteca OpenCV para acessar a câmera e lidar com as matrizes de imagem.
import time # Importa a biblioteca de tempo para criar os cronômetros de preparação e gravação.
import analise_espectral # Importa o SEU módulo de análise que fará o processamento dos canais RGB.

# ==========================================
# PARÂMETROS DO TESTE SEM ARDUINO
# ==========================================
# Define quantos segundos você terá para ligar a lanterna e simular a excitação da amostra.
tempo_preparacao_segundos = 5  
# Define a duração da gravação principal onde ocorrerá o decaimento.
tempo_gravacao_segundos = 15   

# 1 indica a câmera USB secundária (a sua Logitech). 0 abriria a webcam do próprio notebook.
indice_camera = 1 
fps_desejado = 30 # Define a meta de capturar 30 quadros por segundo.
largura = 640 # Largura da resolução do vídeo.
altura = 480 # Altura da resolução do vídeo.

nome_arquivo_teste = 'video_teste_integracao.avi' # Nome do arquivo de vídeo que será gerado.

# ==========================================
# 1. INICIALIZAR CÂMERA
# ==========================================
print("Inicializando Câmera...")
# Abre a conexão com a Logitech forçando a API DirectShow do Windows para evitar instabilidades de driver.
cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

# Define o formato de compressão MJPG para o fluxo de dados que sai da câmera e vai para o cabo USB.
fourcc_cam = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cam) # Aplica essa compressão no hardware da câmera.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura) # Trava a largura geométrica.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura) # Trava a altura geométrica.
cap.set(cv2.CAP_PROP_FPS, fps_desejado) # Pede ao sensor que trabalhe a 30 FPS.

# Travando parâmetros automáticos para o modo manual (Essencial para espectroscopia)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Desliga a compensação automática de luz.
cap.set(cv2.CAP_PROP_EXPOSURE, -5) # Define um nível fixo e manual para a abertura do obturador.
cap.set(cv2.CAP_PROP_AUTO_WB, 0) # Desliga a adaptação automática de cores.
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000) # Fixa os níveis de RGB para a temperatura de 4000 Kelvin.
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Impede que a lente fique embaçando e desembaçando sozinha.
cap.set(cv2.CAP_PROP_FOCUS, 0) # Trava a lente em um ponto de foco fixo.

# Configuração EXTREMAMENTE importante: a variável 0 significa UNCOMPRESSED (Sem compressão).
# Isso garante que o OpenCV salve as matrizes puras (DIB) no disco, sem alterar nenhuma intensidade de cor (lossless).
fourcc_out = 0
# Prepara o gravador de vídeo (out) com o nome, formato cru (0), taxa e dimensões.
out = cv2.VideoWriter(nome_arquivo_teste, fourcc_out, fps_desejado, (largura, altura))

# ==========================================
# 2. ETAPA DE PREPARAÇÃO (SIMULAÇÃO DOS LEDS LIGADOS)
# ==========================================
print(f"\n--- Preparação: Ligue a lanterna do celular! ({tempo_preparacao_segundos}s) ---")
# Marca a hora de início desta fase inicial no relógio do PC.
tempo_inicio_prep = time.time()

# Loop de espera que substitui o acionamento do Arduino. Fica rodando até dar os 5 segundos.
while (time.time() - tempo_inicio_prep) < tempo_preparacao_segundos:
    # Lemos os frames da câmera constantemente apenas para limpar o buffer USB (descartando imagens antigas).
    ret, frame = cap.read() 
    if not ret: # Se a câmera der erro, aborta o loop.
        break
    
    # Desenha um texto laranja pedindo para você apontar a lanterna (fase de simulação de excitação).
    cv2.putText(frame, "LIGUE A LANTERNA...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
    # Mostra a imagem na tela do seu computador.
    cv2.imshow('Teste de Integracao', frame)
    # Aguarda 1 milissegundo para dar tempo de o sistema operacional desenhar a janela.
    cv2.waitKey(1)

# ==========================================
# 3. ETAPA DE GRAVAÇÃO (SIMULAÇÃO DO DECAIMENTO)
# ==========================================
print(f"\n--- Gravando: Apague a lanterna agora! ({tempo_gravacao_segundos}s) ---")
# Zera o cronômetro para iniciar a contagem dos 15 segundos oficiais de gravação.
tempo_inicio_gravacao = time.time()
prev_frame_time = 0 # Inicia a variável auxiliar de tempo para o cálculo do FPS.

# Inicia o loop infinito que fará a coleta oficial dos dados no escuro.
while True:
    ret, frame = cap.read() # Captura a foto exata daquele milissegundo.
    if not ret: # Trava de segurança contra desconexão.
        break
    
    # Calcula quantos segundos se passaram desde que você apagou a lanterna.
    tempo_decorrido = time.time() - tempo_inicio_gravacao
    
    # Se bater o limite dos 15 segundos, interrompe o loop de gravação.
    if tempo_decorrido >= tempo_gravacao_segundos:
        print("\nGravação concluída.")
        break
    
    # SALVAMENTO PURO: Escreve a matriz de pixels limpa e sem compressão no arquivo de vídeo.
    out.write(frame)
    
    # --- MATEMÁTICA DO FPS ---
    new_frame_time = time.time() # Pega a marca de tempo exata desta leitura.
    try:
        # FPS = 1 segundo dividido pelo tempo que demorou entre a foto passada e a atual.
        fps_real = 1 / (new_frame_time - prev_frame_time)
    except ZeroDivisionError:
        fps_real = 0 # Previne erro matemático fatal se a diferença for lida como zero.
    prev_frame_time = new_frame_time # O tempo atual vira o "tempo passado" para a próxima volta do loop.
    
    # Calcula quantos segundos faltam. A função max(0, ...) evita que apareça um tempo negativo bizarro.
    tempo_restante = max(0, tempo_gravacao_segundos - tempo_decorrido)
    
    # Desenha as interfaces de texto na imagem para guiar o seu teste:
    # Aviso em vermelho para apagar a luz.
    cv2.putText(frame, "APAGUE A LANTERNA", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Contador de FPS em verde.
    cv2.putText(frame, f"FPS: {int(fps_real)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Cronômetro regressivo em branco.
    cv2.putText(frame, f"Tempo: {tempo_restante:.1f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostra a imagem atualizada com os textos.
    cv2.imshow('Teste de Integracao', frame)
    
    # Comando de pânico: se apertar 'q', quebra o loop e aborta a gravação antecipadamente.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProcesso interrompido manualmente.")
        break

# ==========================================
# 4. FINALIZAÇÃO E CHAMADA DO MÓDULO DE ANÁLISE
# ==========================================
# Desconecta a câmera da porta USB do PC.
cap.release()
# Fecha o contêiner de vídeo, salvando os dados no HD de forma segura.
out.release()
# Destrói as janelas de monitoramento abertas na área de trabalho.
cv2.destroyAllWindows()
print(f"Vídeo de teste salvo como '{nome_arquivo_teste}'")

print("\n--- Iniciando Módulo de Análise Espectral ---")
# O módulo vai abrir a janela pedindo para você confirmar ou selecionar a ROI.
# O Python passa o nome do arquivo que acabamos de gravar diretamente para o seu script de análise trabalhar.
analise_espectral.executar_analise(nome_arquivo_teste)
