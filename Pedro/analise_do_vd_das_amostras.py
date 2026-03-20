import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys # Importado para permitir o encerramento limpo com ESC
import pandas as pd # Adicionado para criar o arquivo CSV de forma organizada
import os # Adicionado para garantir o caminho da pasta do script

def selecionar_roi_manual(video_path, frame_ref):
    """Abre o frame de referência para seleção manual do quadrado."""
    # Criar janela redimensionável para não estourar a tela caso o vídeo seja 4K/FullHD
    window_name = "Selecione a ROI Manualmente (Desenhe o quadrado)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # AJUSTE DE RESOLUÇÃO: 1280x800
    cv2.resizeWindow(window_name, 1280, 800)
    
    # Abre o seletor ROI do OpenCV sobre o frame de referência (já excitado)
    roi = cv2.selectROI(window_name, frame_ref, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name) # Fecha apenas esta janela
    
    # x, y são as coordenadas do canto superior esquerdo; w, h são largura e altura
    return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

def encontrar_frame_ideal_por_tempo(video_path, threshold_queda=235, voltar_n_segundos=1.0):
    """
    Analisa o vídeo para encontrar o momento em que a luz apaga e retrocede 
    N segundos para encontrar o frame ideal de detecção da amostra.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        return None, 0
    
    frame_idx = 0
    ponto_queda_frame = 0
    encontrou_excitacao = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Monitora o brilho máximo (ponto mais saturado do frame)
        max_val = np.max(frame)
        
        # Estado 1: Procurando o momento em que a luz UV é ligada
        if not encontrou_excitacao and max_val >= threshold_queda:
            encontrou_excitacao = True
            
        # Estado 2: Já excitou, agora procura o momento que a luz apagou (caiu do threshold)
        if encontrou_excitacao and max_val < threshold_queda:
            ponto_queda_frame = frame_idx
            break
        frame_idx += 1
    
    # Calcula quantos frames voltar para pegar a amostra ainda acesa e estável
    # Sincroniza o tempo (segundos) com a taxa de quadros (FPS) do vídeo
    frames_para_voltar = int(voltar_n_segundos * fps)
    frame_alvo_idx = max(0, ponto_queda_frame - frames_para_voltar)
    
    # Pula o vídeo para o frame alvo e captura a imagem
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_alvo_idx)
    ret, frame_final = cap.read()
    cap.release()
    
    # Retorna o frame de referência e seu índice temporal
    return frame_final, frame_alvo_idx

def detectar_amostra_automatica(frame, z_margem=10):
    """
    Tenta detectar automaticamente a amostra fluorescente no frame de referência.
    Retorna a ROI detectada ou None se falhar.
    """
    if frame is None: return None
    
    # Pré-processamento simples para destacar a amostra contra o fundo escuro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Cria uma máscara binária: tudo acima de 50 fica branco, resto preto
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # Encontra os contornos das áreas brancas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Assume que a amostra é o maior contorno detectado
    maior_contorno = max(contours, key=cv2.contourArea)
    
    # Encontra o círculo mínimo que circunscreve a amostra
    (x_c, y_c), raio = cv2.minEnclosingCircle(maior_contorno)
    
    # Cálculo Geométrico do Quadrado Inscrito:
    # Lado do quadrado = (Raio * sqrt(2)) - (2 * Margem de Segurança Z)
    # Isso garante que a área de média esteja bem centralizada na resina
    lado = int(raio * 1.4142) - (2 * z_margem)
    
    if lado <= 0:
        return None
    
    # Calcula as coordenadas do canto superior esquerdo do quadrado
    x = max(0, int(x_c - (lado / 2)))
    y = max(0, int(y_c - (lado / 2)))
    
    # Retorna as coordenadas e dimensões para o ROI do OpenCV [x, y, deltaX, deltaY]
    return [x, y, lado, lado]

def validar_roi_interativamente(frame, roi_auto, threshold_queda, voltar_n_segundos):
    """
    Apresenta o ROI automático na tela para conferência do usuário.
    Permite aceitar (ENTER), ir para manual ('M') ou cancelar (ESC).
    """
    if roi_auto is None: return None # Proteção se o automático já tiver falhado internamente
    
    x, y, lado, _ = roi_auto
    frame_viz = frame.copy()
    
    # Desenha o quadrado da ROI (Verde, espessura 3)
    cv2.rectangle(frame_viz, (x, y), (x+lado, y+lado), (0, 255, 0), 3)
    
    # Adiciona texto explicativo no frame
    msg_titulo = f"Conferencia ROI - Frame Ideal (T-{voltar_n_segundos}s)"
    msg_instrucoes = "ENTER: Aceitar | 'M': Mudar para Manual | ESC: Cancelar"
    
    cv2.putText(frame_viz, msg_titulo, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_viz, msg_instrucoes, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    window_name = "Conferência ROI Automático"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # AJUSTE DE RESOLUÇÃO: 1280x800
    cv2.resizeWindow(window_name, 1280, 800) 
    cv2.imshow(window_name, frame_viz)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == 32: # ENTER ou ESPAÇO
            cv2.destroyWindow(window_name)
            return roi_auto
        elif key == ord('m') or key == ord('M'): # Mudar para manual
            cv2.destroyWindow(window_name)
            return None 
        elif key == 27: # ESC
            cv2.destroyAllWindows()
            sys.exit()

def analisar_video_puro(video_path, roi, frame_inicial, fps_alvo=5):
    """
    Analisa o vídeo extraindo a média RGB pura da região especificada (ROI),
    a partir de um frame inicial, sem aplicar normalizações ou filtros.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    intervalo = max(1, int(fps_orig / fps_alvo))
    x, y, w, h = roi
    
    # Inicia a leitura do vídeo exatamente no frame de recorte temporal (voltar_n_segundos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)
    
    tempos, dados_rgb = [], []
    curr_frame = frame_inicial
    
    print(f"Analisando ROI ({w}x{h} pixels)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Processa apenas nos frames de intervalo para atingir o fps_alvo
        if (curr_frame - frame_inicial) % intervalo == 0:
            crop = frame[y:y+h, x:x+w]
            
            if crop.size > 0:
                # DADOS PUROS: Extração direta canal por canal usando float64.
                media_r = np.mean(crop[:, :, 2].astype(np.float64))
                media_g = np.mean(crop[:, :, 1].astype(np.float64))
                media_b = np.mean(crop[:, :, 0].astype(np.float64))
                
                dados_rgb.append([media_r, media_g, media_b])
                tempos.append(curr_frame / fps_orig)
        
        curr_frame += 1
        
    cap.release()
    return np.array(tempos), np.array(dados_rgb)

# ==========================================
# PARÂMETROS DE CONTROLE (AJUSTE AQUI)
# ==========================================
video_file = Path("c:/Users/Micro/Documents/videos_amostras/Primeira rodada-20260320T213054Z-3-001/Primeira rodada/Azul_151535.mp4")

z_margem_seguranca = 40 
voltar_n_segundos = 15.0 
threshold_brilho = 235   
fps_analise = 5          
modo_manual_forcado = False 
conferir_roi_interativamente = False 

# ==========================================
# LÓGICA PRINCIPAL DE EXECUÇÃO
# ==========================================
if video_file.exists():
    try:
        # 1. Busca o momento exato de desligamento e retrocede o tempo para o frame ideal
        frame_ref, idx_start = encontrar_frame_ideal_por_tempo(video_file, threshold_brilho, voltar_n_segundos)
        
        if frame_ref is None:
            raise Exception("Erro na leitura do vídeo.")

        # 2. Define a ROI (Automática ou Manual de Backup)
        roi_final = None
        
        if not modo_manual_forcado:
            roi_auto_tentativa = detectar_amostra_automatica(frame_ref, z_margem=z_margem_seguranca)
            
            if conferir_roi_interativamente:
                roi_final = validar_roi_interativamente(frame_ref, roi_auto_tentativa, threshold_brilho, voltar_n_segundos)
            else:
                roi_final = roi_auto_tentativa
            
        if roi_final is None:
            roi_final = selecionar_roi_manual(video_file, frame_ref)

        # 3. Processamento Final: Analisa os dados puros
        if roi_final and roi_final[2] > 0:
            t, rgb = analisar_video_puro(video_file, roi_final, idx_start, fps_alvo=fps_analise)
            
            # --- DEFININDO PASTA DO SCRIPT ---
            pasta_do_script = Path(os.getcwd()) # Pega o local onde o script está rodando
            nome_base = video_file.stem # Pega o nome do vídeo sem a extensão .mp4
            
            # 4. Exportação de Dados para CSV
            t_relativo = t - t[0]
            df_export = pd.DataFrame({
                'Tempo_s': t_relativo,
                'R_Puro': rgb[:, 0],
                'G_Puro': rgb[:, 1],
                'B_Puro': rgb[:, 2],
                'Media_Total': np.mean(rgb, axis=1)
            })
            
            # Define o salvamento na pasta do script (Portabilidade Máxima)
            csv_output = pasta_do_script / f"{nome_base}.csv"
            df_export.to_csv(csv_output, index=False)
            print(f"Dados exportados na pasta do script (CSV): {csv_output}")

            # 5. Plotagem Científica
            plt.figure(figsize=(12, 6))
            plt.plot(t_relativo, rgb[:, 0], 'red', label='Canal R (Vermelho)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 1], 'green', label='Canal G (Verde)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 2], 'blue', label='Canal B (Azul)', linewidth=1.5)
            plt.plot(t_relativo, np.mean(rgb, axis=1), 'black', linestyle='--', label='Média Total (Escala de Cinza)', alpha=0.7)

            plt.title(f"Cinética de Luminescência: {video_file.name}")
            plt.xlabel("Tempo Relativo (s)")
            plt.ylabel("Intensidade Bruta (0-255)")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.5)
            
            # Define o salvamento do gráfico de referência na pasta do script
            graph_output = pasta_do_script / f"{nome_base}.png"
            plt.savefig(graph_output, dpi=300)
            print(f"Gráfico de referência salvo: {graph_output}")
            
            plt.show()
        else:
            print("Análise cancelada: ROI não definido.")

    except Exception as e:
        print(f"Erro: {e}")
        cv2.destroyAllWindows()
else:
    print(f"Erro: Arquivo não encontrado.")
