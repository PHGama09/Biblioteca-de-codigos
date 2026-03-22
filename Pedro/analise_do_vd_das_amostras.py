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

def localizar_queda_na_roi(video_path, roi, threshold_queda=180):
    """
    Analisa apenas a área da amostra (ROI) para encontrar o momento exato 
    em que a intensidade cai, ignorando reflexos externos no vídeo.
    """
    cap = cv2.VideoCapture(str(video_path))
    x, y, w, h = roi
    frame_idx = 0
    ponto_queda_frame = 0
    encontrou_excitacao = False

    print(f"Localizando queda real na ROI (Threshold: {threshold_queda})...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Foca a análise apenas na região da amostra selecionada
        roi_focus = frame[y:y+h, x:x+w]
        brilho_medio = np.mean(roi_focus) # Média de brilho interna da amostra

        if not encontrou_excitacao and brilho_medio >= threshold_queda:
            encontrou_excitacao = True
            
        if encontrou_excitacao and brilho_medio < threshold_queda:
            ponto_queda_frame = frame_idx
            break
        frame_idx += 1
    
    cap.release()
    return ponto_queda_frame

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
    msg_titulo = f"Conferencia ROI - Amostra Detectada"
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
            print("Execução cancelada pelo usuário (ESC).")
            sys.exit()

def analisar_video_puro(video_path, roi, frame_inicial, fps_alvo=5, duracao_max_s=None):
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
    
    # Calcula o frame limite se duracao_max_s for definido
    frame_limite = None
    if duracao_max_s is not None:
        frame_limite = frame_inicial + int(duracao_max_s * fps_orig)

    print(f"Analisando ROI ({w}x{h} pixels)...")
    
    while cap.isOpened():
        # Verifica se atingiu o tempo limite definido pelo usuário
        if frame_limite is not None and curr_frame > frame_limite:
            break

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
                # Tempo relativo para o gráfico
                tempos.append((curr_frame - frame_inicial) / fps_orig)
        
        curr_frame += 1
        
    cap.release()

    # --- EQUALIZAÇÃO PARA O PONTO FINAL ---
    if duracao_max_s is not None and len(tempos) > 0 and tempos[-1] < duracao_max_s:
        tempos.append(duracao_max_s)
        dados_rgb.append(dados_rgb[-1])

    return np.array(tempos), np.array(dados_rgb)

# ==========================================
# PARÂMETROS DE CONTROLE (AJUSTE AQUI)
# ==========================================
caminho_original = r"c:\Users\Micro\Documents\videos_amostras\Primeira_Rodada-20260320T213054Z-3-001\Azul_151535.mp4"
video_file = Path(caminho_original.replace("\\", "/")) #replace para n precisar mudar manualmente o \ da linha de caminho do dwindows para / como um caminho reconhecivel
# Measure-Command { python   analise_do_vd_das_amostras.py }

z_margem_seguranca = 100
segundo_referencia_roi = 10.0 # Define em qual segundo do vídeo a amostra já está acesa para capturar a ROI
voltar_n_segundos = 20.0 
duracao_analise_segundos = 60.0 + voltar_n_segundos # Define quantos segundos de vídeo serão analisados após o ponto inicial
threshold_brilho = 180   
fps_analise = 30           
modo_manual_forcado = False 
conferir_roi_interativamente = True

# ==========================================
# LÓGICA PRINCIPAL DE EXECUÇÃO
# ==========================================
if video_file.exists():
    try:
        # 1. Busca o frame de referência para a ROI baseado no tempo definido
        cap_temp = cv2.VideoCapture(str(video_file))
        fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
        frame_referencia_idx = int(segundo_referencia_roi * fps_temp)
        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_referencia_idx) 
        ret_temp, frame_ref = cap_temp.read() # Corrigido: desempacotamento seguro do retorno
        cap_temp.release()
        
        if not ret_temp or frame_ref is None:
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

        # 3. Localiza a queda real baseada APENAS na ROI e calcula o início da análise
        idx_queda = localizar_queda_na_roi(video_file, roi_final, threshold_brilho)
        fps = cv2.VideoCapture(str(video_file)).get(cv2.CAP_PROP_FPS)
        idx_start = max(0, idx_queda - int(voltar_n_segundos * fps))

        # 4. Processamento Final: Analisa os dados puros
        if roi_final and roi_final[2] > 0:
            t_relativo, rgb = analisar_video_puro(video_file, roi_final, idx_start, fps_alvo=fps_analise, duracao_max_s=duracao_analise_segundos)
            
            # --- DEFININDO PASTA DO SCRIPT ---
            pasta_do_script = Path(os.getcwd()) # Pega o local onde o script está rodando
            nome_base = video_file.stem # Pega o nome do vídeo sem a extensão .mp4
            
            # 5. Exportação de Dados para CSV
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

            # 6. Plotagem Científica
            plt.figure(figsize=(12, 6))
            plt.plot(t_relativo, rgb[:, 0], 'red', label='Canal R (Vermelho)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 1], 'green', label='Canal G (Verde)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 2], 'blue', label='Canal B (Azul)', linewidth=1.5)
            plt.plot(t_relativo, np.mean(rgb, axis=1), 'black', linestyle='--', label='Média Total (Escala de Cinza)', alpha=0.7)
            
            # Linha indicando o momento exato detectado do desligamento
            plt.axvline(x=voltar_n_segundos, color='gray', linestyle=':', label='Desligamento Detectado')

            plt.title(f"Análise do decaimento de Luminescência: {video_file.name}")
            plt.xlabel("Tempo Relativo (s)")
            plt.ylabel("Intensidade Bruta (0-255)")
            
            # --- FORÇANDO EQUALIZAÇÃO VISUAL ---
            plt.xlim(0, duracao_analise_segundos)
            plt.gca().set_xmargin(0)
            
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

# Measure-Command { python seu_arquivo.py } comando para checar o tempo
# Measure-Command { python   analise_do_vd_das_amostras.py }
