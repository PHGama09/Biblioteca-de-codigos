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

def detectar_amostra_automatica(frame, lado_definido=120, threshold_bin=200, auto_lado=False, margem_z=10, lado_min=10):
    """
    Tenta detectar automaticamente a amostra fluorescente usando Momentos de Imagem.
    Calcula o ROI centralizado. Se auto_lado for True, ignora lado_definido e calcula inscrito - margem_z.
    """
    if frame is None: return None
    
    # Pré-processamento conforme o código 1: Escala de cinza e Limiarização
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_bin, 255, cv2.THRESH_BINARY)
    
    # Cálculo de Momentos para achar o centroide (cx, cy)
    M = cv2.moments(thresh)
    if M["m00"] == 0: return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Lógica de definição do lado
    if auto_lado:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        maior_contorno = max(contours, key=cv2.contourArea)
        _, raio = cv2.minEnclosingCircle(maior_contorno)
        # Lado do quadrado inscrito = (Raio * sqrt(2)) - (2 * Margem Z)
        lado = max(lado_min, int(int(raio * 1.4142) - (2 * margem_z)))
    else:
        lado = lado_definido

    # Calcula coordenadas (x, y) do canto superior esquerdo
    tam = lado // 2
    x = max(0, cx - tam)
    y = max(0, cy - tam)
    
    return [x, y, lado, lado]

def validar_roi_interativamente(frame, roi_auto, threshold_queda, voltar_n_segundos):
    """
    Apresenta o ROI automático na tela para conferência do usuário.
    Permite aceitar (ENTER), ir para manual ('M') ou cancelar (ESC).
    """
    if roi_auto is None: return None # Proteção se o automático já tiver falhado internamente
    
    x, y, w, h = roi_auto
    frame_viz = frame.copy()
    
    # Desenha o quadrado da ROI (Verde, espessura 3) e exibe métrica
    cv2.rectangle(frame_viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(frame_viz, f"ROI: {w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
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
caminho_original = r"c:\Users\Micro\Documents\videos_amostras\Segunda_Rodada-20260322T213739Z-3-001\Segunda_Rodada\Vermelho_152634.mp4"
video_file = Path(caminho_original.replace("\\", "/")) 
# Measure-Command { python analise_do_vd_das_amostras.py }

# NOVOS PARÂMETROS SOLICITADOS:
usar_metodo_automatico = True        # True para tentar o automático primeiro, False para ir direto ao manual
calcular_lado_automatico = True     # NOVO: True para o script calcular o quadrado inscrito sozinho usando Z_MARGEM
lado_roi_manual = 120                # Usado apenas se calcular_lado_automatico = False
salvar_frame_roi = True              # True para salvar a imagem do ROI antes de iniciar a análise
threshold_binarizacao = 200          # Limiar para achar o centro da amostra (código 1 usava 200)

z_margem_seguranca = 50     # Margem para garantir que o ROI automático esteja dentro da amostra
lado_minimo_roi = 20         # NOVO: Garante que o lado do ROI nunca seja menor que este valor ou negativo
segundo_referencia_roi = 10.0 
voltar_n_segundos = 20.0 
duracao_analise_segundos = 60.0 + voltar_n_segundos 
threshold_brilho = 180   
fps_analise = 30           
modo_manual_forcado = not usar_metodo_automatico 
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
        ret_temp, frame_ref = cap_temp.read() 
        cap_temp.release()
        
        if not ret_temp or frame_ref is None:
            raise Exception("Erro na leitura do vídeo.")

        # 2. Define a ROI (Automática ou Manual de Backup)
        roi_final = None
        
        if not modo_manual_forcado:
            # Chama a detecção com as novas flags de automação de lado e margem Z
            roi_auto_tentativa = detectar_amostra_automatica(frame_ref, 
                                                            lado_definido=lado_roi_manual, 
                                                            threshold_bin=threshold_binarizacao,
                                                            auto_lado=calcular_lado_automatico,
                                                            margem_z=z_margem_seguranca,
                                                            lado_min=lado_minimo_roi)
            
            if conferir_roi_interativamente:
                roi_final = validar_roi_interativamente(frame_ref, roi_auto_tentativa, threshold_brilho, voltar_n_segundos)
            else:
                roi_final = roi_auto_tentativa
            
        # Se o automático falhou ou o usuário recusou no modo interativo, abre o manual
        if roi_final is None:
            roi_final = selecionar_roi_manual(video_file, frame_ref)

        # SALVAMENTO DO FRAME DO ROI (Se ativado)
        if salvar_frame_roi and roi_final:
            x, y, w, h = roi_final
            frame_salvamento = frame_ref.copy()
            cv2.rectangle(frame_salvamento, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame_salvamento, f"ROI: {w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(Path(os.getcwd()) / f"{video_file.stem}_ROI_Config.png"), frame_salvamento)
            print(f"Frame de ROI salvo em: {os.getcwd()}")

        # 3. Localiza a queda real baseada APENAS na ROI e calcula o início da análise
        idx_queda = localizar_queda_na_roi(video_file, roi_final, threshold_brilho)
        fps = cv2.VideoCapture(str(video_file)).get(cv2.CAP_PROP_FPS)
        idx_start = max(0, idx_queda - int(voltar_n_segundos * fps))

        # 4. Processamento Final: Analisa os dados puros
        if roi_final and roi_final[2] > 0:
            t_relativo, rgb = analisar_video_puro(video_file, roi_final, idx_start, fps_alvo=fps_analise, duracao_max_s=duracao_analise_segundos)
            
            # --- DEFININDO PASTA DO SCRIPT ---
            pasta_do_script = Path(os.getcwd()) 
            nome_base = video_file.stem 
            
            # 5. Exportação de Dados para CSV
            df_export = pd.DataFrame({
                'Tempo_s': t_relativo,
                'R_Puro': rgb[:, 0],
                'G_Puro': rgb[:, 1],
                'B_Puro': rgb[:, 2],
                'Media_Total': np.mean(rgb, axis=1)
            })
            
            csv_output = pasta_do_script / f"{nome_base}.csv"
            df_export.to_csv(csv_output, index=False)
            print(f"Dados exportados na pasta do script (CSV): {csv_output}")

            # 6. Plotagem Científica
            plt.figure(figsize=(12, 6))
            plt.plot(t_relativo, rgb[:, 0], 'red', label='Canal R (Vermelho)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 1], 'green', label='Canal G (Verde)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 2], 'blue', label='Canal B (Azul)', linewidth=1.5)
            plt.plot(t_relativo, np.mean(rgb, axis=1), 'black', linestyle='--', label='Média Total (Escala de Cinza)', alpha=0.7)
            
            plt.axvline(x=voltar_n_segundos, color='gray', linestyle=':', label='Desligamento Detectado')

            plt.title(f"Análise do decaimento de Luminescência: {video_file.name}")
            plt.xlabel("Tempo Relativo (s)")
            plt.ylabel("Intensidade Bruta (0-255)")
            
            plt.xlim(0, duracao_analise_segundos)
            plt.gca().set_xmargin(0)
            
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.5)
            
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

# Measure-Command { python analise_do_vd_das_amostras.py }
# Measure-Command { python seu_arquivo.py } Comando para cronometar o tempo de compilação
