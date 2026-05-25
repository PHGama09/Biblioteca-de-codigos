import cv2 # Importa a biblioteca OpenCV para controle da câmera e interface gráfica.

# Função "vazia" necessária para a criação dos botões deslizantes (trackbars) no OpenCV.
# O OpenCV exige que você passe uma função que será chamada toda vez que o botão for mexido, mesmo que não faça nada.
def nada(x):
    pass

# Inicializa a conexão com a câmera USB (Logitech) usando a API DirectShow do Windows.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Define uma string com o nome da janela principal.
janela = "Painel de Calibracao Logitech"

# Cria a janela antes de abri-la e permite que ela seja redimensionável (WINDOW_NORMAL).
cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

# Redimensiona a janela para um tamanho maior (800x800) para garantir que caibam todos os botões deslizantes na tela.
cv2.resizeWindow(janela, 800, 800)

# ==========================================
# CRIAÇÃO DOS BOTÕES DESLIZANTES (TRACKBARS)
# Parâmetros: Nome na tela, Janela onde vai ficar, Valor Inicial, Valor Máximo, Função ativada ao mexer.
# IMPORTANTE: Trackbars só aceitam números inteiros positivos (0, 1, 2...). 
# Se precisarmos de decimais ou negativos, fazemos a conversão matemática lá no loop.
# ==========================================

cv2.createTrackbar("Auto Exp (0=Ligado, 25=Desl)", janela, 25, 100, nada) 
cv2.createTrackbar("Exposicao (-10 a 0)", janela, 5, 10, nada) # Vai de 0 a 10 (depois subtraímos 10 para virar -10 a 0).
cv2.createTrackbar("Auto WB (0=Desl, 1=Ligado)", janela, 0, 1, nada) # Balanço de Branco Automático (0 ou 1).
cv2.createTrackbar("Temp Cor (2000-10000K)", janela, 2000, 8000, nada) # Vai de 0 a 8000 (somamos 2000 no loop para virar 2000K a 10000K).
cv2.createTrackbar("Auto Foco (0=Desl, 1=Ligado)", janela, 0, 1, nada) # Foco Automático (0 ou 1).
cv2.createTrackbar("Foco Manual (0 a 255)", janela, 0, 255, nada) # Controle físico do motor da lente.
cv2.createTrackbar("Brilho (0 a 255)", janela, 128, 255, nada) # Luminosidade digital geral.
cv2.createTrackbar("Contraste (0 a 255)", janela, 128, 255, nada) # Diferença entre claro e escuro.
cv2.createTrackbar("Saturacao (0 a 255)", janela, 128, 255, nada) # Intensidade das cores (0 = Preto e Branco).
cv2.createTrackbar("Ganho (0 a 255)", janela, 0, 255, nada) # Sensibilidade ISO digital do sensor.

print("--- GUIA DE CALIBRAÇÃO AVANÇADA ---")
print("1. Ajuste os valores na tela.")
print("2. Quando encontrar a imagem perfeita, anote os valores para colocar no script de coleta.")
print("3. Aperte 'q' para sair.")

# Loop infinito onde aplicamos os valores em tempo real.
while True:
    # Captura a foto da câmera para mostrar na janela.
    ret, frame = cap.read()
    if not ret:
        break # Aborta se a câmera falhar.
    
    # ==========================================
    # LEITURA DOS VALORES ATUAIS DOS BOTÕES
    # ==========================================
    valor_auto_exp = cv2.getTrackbarPos("Auto Exp (0=Ligado, 25=Desl)", janela)
    valor_exp = cv2.getTrackbarPos("Exposicao (-10 a 0)", janela)
    valor_auto_wb = cv2.getTrackbarPos("Auto WB (0=Desl, 1=Ligado)", janela)
    valor_wb = cv2.getTrackbarPos("Temp Cor (2000-10000K)", janela)
    valor_auto_foco = cv2.getTrackbarPos("Auto Foco (0=Desl, 1=Ligado)", janela)
    valor_foco = cv2.getTrackbarPos("Foco Manual (0 a 255)", janela)
    valor_brilho = cv2.getTrackbarPos("Brilho (0 a 255)", janela)
    valor_contraste = cv2.getTrackbarPos("Contraste (0 a 255)", janela)
    valor_saturacao = cv2.getTrackbarPos("Saturacao (0 a 255)", janela)
    valor_ganho = cv2.getTrackbarPos("Ganho (0 a 255)", janela)
    
    # ==========================================
    # CONVERSÃO MATEMÁTICA PARA A CÂMERA
    # ==========================================
    # Converte o inteiro para decimal (Ex: 25 vira 0.25).
    auto_exp_real = valor_auto_exp / 100.0 
    # Converte o range de 0 a 10 para -10 a 0.
    exp_real = valor_exp - 10 
    # Adiciona 2000 para converter o slider de (0 a 8000) para Kelvin real (2000 a 10000).
    wb_real = valor_wb + 2000 
    
    # ==========================================
    # ENVIO DOS COMANDOS PARA O HARDWARE DA CÂMERA
    # ==========================================
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_real)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp_real)
    cap.set(cv2.CAP_PROP_AUTO_WB, valor_auto_wb)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_real)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, valor_auto_foco)
    cap.set(cv2.CAP_PROP_FOCUS, valor_foco)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, valor_brilho)
    cap.set(cv2.CAP_PROP_CONTRAST, valor_contraste)
    cap.set(cv2.CAP_PROP_SATURATION, valor_saturacao)
    cap.set(cv2.CAP_PROP_GAIN, valor_ganho)
    
    # Textos removidos da imagem para evitar poluição visual, já que o próprio Windows 
    # mostra o número exato ao lado de cada botão deslizante na interface gráfica.
    
    # Exibe a imagem já afetada pelas alterações do sensor.
    cv2.imshow(janela, frame)
    
    # Espera 1 milissegundo. Se a tecla 'q' for pressionada, quebra o loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera.
cap.release()
# Destrói a interface gráfica.
cv2.destroyAllWindows()
