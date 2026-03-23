import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#TIPO RESINA -----------------------------------------------------
#Coloque aqui a cor da resina analisada e a rodada correspondente
cor = "Verde"
rodada = "3"

# CONFIGURAÇÕES DE CAMINHO ---------------------------------------------------
# 1. Onde está o vídeo original?
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
video_nome = "Azul_153241.mp4"
video_path = os.path.join(downloads_path, video_nome)

# 2. Onde você quer salvar os resultados? (Cole o caminho entre as aspas)
# Exemplo: pasta_salvamento = r"C:\Resultados\ProjetoResina"
pasta_salvamento = rf"C:\Users\imene\Downloads\Pyhton_GLIF\{cor}_{rodada}"

# Cria a pasta caso ela não exista
if not os.path.exists(pasta_salvamento):
    os.makedirs(pasta_salvamento)
    print(f"Pasta criada: {pasta_salvamento}")

nome_base = video_nome.split('.')[0]

# PARÂMETROS -----------------------------------------------------------------
intervalo = 0.034
limiar_trigger = 252
t_pre = 20           
t_pos = 60           

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 1. AUTO-ROI E VISUALIZAÇÃO -------------------------------------------------
cap.set(cv2.CAP_PROP_POS_MSEC, 10000) 
ret, frame_ref = cap.read()

if ret:
    gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (320, 240)
    
    tam = 60
    y1, y2, x1, x2 = cy-tam, cy+tam, cx-tam, cx+tam
    
    # Desenha o ROI no frame para visualização
    frame_visual = frame_ref.copy()
    cv2.rectangle(frame_visual, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame_visual, f"ROI: {tam*2}x{tam*2}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Salva a imagem do ROI definido na nova pasta
    cv2.imwrite(os.path.join(pasta_salvamento, f"{cor}_{rodada}_rodada_ROI.png"), frame_visual)
    
    # Mostra a tela com o ROI
    cv2.imshow("Verifique o ROI - Pressione qualquer tecla para continuar", frame_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. ANÁLISE COMPLETA --------------------------------------------------------
data_raw = []
frames_passo = int(intervalo * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f"Analisando vídeo... Salvando em: {pasta_salvamento}")
idx_trigger = None
frame_atual = 0

while frame_atual < total_frames:
    ret, frame = cap.read()
    if not ret: break

    roi = frame[y1:y2, x1:x2]
    b, g, r = np.mean(roi, axis=(0, 1))
    intensidade = (r + g + b) / 3
    tempo_vidal_real = frame_atual / fps
    
    data_raw.append([tempo_vidal_real, r, g, b, intensidade])

    if idx_trigger is None and len(data_raw) > 1:
        if data_raw[-2][4] >= limiar_trigger and intensidade < limiar_trigger:
            idx_trigger = len(data_raw) - 1
            print(f"Trigger detectado em: {tempo_vidal_real:.3f}s")

    if idx_trigger is not None:
        if tempo_vidal_real > (data_raw[idx_trigger][0] + t_pos + 2):
            break

    for _ in range(frames_passo - 1): cap.grab()
    frame_atual += frames_passo

cap.release()

# 3. FILTRAGEM, METADADOS E SALVAMENTO ---------------------------------------
if idx_trigger is not None:
    t_trigger_original = data_raw[idx_trigger][0]
    limite_inferior = t_trigger_original - t_pre
    limite_superior = t_trigger_original + t_pos
    
    data_final = []
    for linha in data_raw:
        if limite_inferior <= linha[0] <= limite_superior:
            data_final.append([linha[0] - limite_inferior, linha[1], linha[2], linha[3], linha[4]])
    
    df = pd.DataFrame(data_final, columns=["Tempo", "R", "G", "B", "Intensidade"])
    
    # Salva CSV de Dados
    df.to_csv(os.path.join(pasta_salvamento, f"{cor}_{rodada}_rodada_analise.csv"), index=False)

    # Criação do arquivo de Parâmetros (Metadados)
    parametros = {
        "Parametro": [
            "Tempo inicial do recorte (s vídeo)", 
            "Ponto do trigger (s vídeo)", 
            "Intervalo de amostragem (s)", 
            "FPS analisados", 
            "R Max", "R Min", 
            "G Max", "G Min", 
            "B Max", "B Min"
        ],
        "Valor": [
            round(limite_inferior, 3),
            round(t_trigger_original, 3),
            intervalo,
            round(fps, 2),
            round(df["R"].max(), 2), round(df["R"].min(), 2),
            round(df["G"].max(), 2), round(df["G"].min(), 2),
            round(df["B"].max(), 2), round(df["B"].min(), 2)
        ]
    }
    df_params = pd.DataFrame(parametros)
    df_params.to_csv(os.path.join(pasta_salvamento, f"{cor}_{rodada}_rodada_parametros.csv"), index=False)

    # 4. PLOT E SALVAMENTO DO GRÁFICO ----------------------------------------
    plt.figure(figsize=(10,6))
    plt.axvline(x=t_pre, color='black', linestyle='--', label='Desligamento UV')
    plt.plot(df["Tempo"], df["R"], color='red', label='Intensidade Vermelho')
    plt.plot(df["Tempo"], df["G"], color='green', label='Intensidade Verde')
    plt.plot(df["Tempo"], df["B"], color='blue', label='Intensidade Azul')
    
    plt.title(f"Arquivo: {cor} - {rodada}° Rodada\nJanela de Excitação ({t_pre}s) e Decaimento ({t_pos}s) - 30 FPS")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Intensidade")
    plt.xlim(0, t_pre + t_pos)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salva o gráfico automaticamente na nova pasta
    plt.savefig(os.path.join(pasta_salvamento, f"{cor}_{rodada}_rodada_grafico.png"))
    plt.show()
    
    print(f"Processamento concluído. 4 arquivos gerados em: {pasta_salvamento}")
else:
    print("Trigger não encontrado. Verifique o limiar_trigger.")
