import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys # Importado para permitir o encerramento limpo com ESC
import pandas as pd # Adicionado para criar o arquivo CSV de forma organizada
import os # Adicionado para garantir o caminho da pasta do script

def selecionar_roi_manual(video_path, frame_ref):
    """Abre o frame de referência para seleção manual do quadrado."""
    print("\n--- SELEÇÃO MANUAL INICIADA ---")
    print("Use o mouse para desenhar um retângulo sobre a área central da amostra luminosa.")
    print("Pressione ENTER ou ESPAÇO para confirmar a seleção.")
    print("Pressione 'c' para cancelar a seleção atual e tentar novamente.")

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
        print("Aviso: Margem Z muito grande para o tamanho da amostra detectada.")
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
    
    # Cria uma cópia do frame para visualização (não altera o original)
    frame_viz = frame.copy()
    
    # Desenha o quadrado da ROI (Verde, espessura 3)
    cv2.rectangle(frame_viz, (x, y), (x+lado, y+lado), (0, 255, 0), 3)
    
    # Adiciona texto explicativo no frame
    msg_titulo = f"Conferencia ROI - Amostra Detectada"
    msg_instrucoes = "ENTER: Aceitar | 'M': Mudar para Manual | ESC: Cancelar"
    
    # Posições do texto no frame (ajustado para Full HD, pode precisar ajuste para 4K)
    cv2.putText(frame_viz, msg_titulo, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_viz, msg_instrucoes, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    window_name = "Conferência ROI Automático"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # AJUSTE DE RESOLUÇÃO: 1280x800
    cv2.resizeWindow(window_name, 1280, 800) 
    cv2.imshow(window_name, frame_viz)
    
    print("\n--- AGUARDANDO VALIDAÇÃO DO ROI ---")
    
    while True:
        # Aguarda tecla pressionada
        key = cv2.waitKey(0) & 0xFF
        
        # ENTER ou ESPAÇO: Aceita o automático
        if key == 13 or key == 32: 
            cv2.destroyWindow(window_name)
            print("ROI automático aceito pelo usuário.")
            return roi_auto
            
        # 'M' ou 'm': Rejeita e vai para manual
        elif key == ord('m') or key == ord('M'):
            cv2.destroyWindow(window_name)
            print("ROI automático rejeitado. Abrindo seletor manual...")
            return None # Retorna None para acionar o fallback manual
            
        # ESC: Cancela tudo
        elif key == 27:
            cv2.destroyAllWindows()
            print("Execução cancelada pelo usuário (ESC).")
            sys.exit() # Encerra o programa Python
            
def analisar_video_puro(video_path, roi, frame_inicial, fps_alvo=5, tempo_limite_obs=None):
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
    
    print(f"\nIniciando análise RGB pura a partir do frame {frame_inicial}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        tempo_relativo_atual = (curr_frame - frame_inicial) / fps_orig
        
        # Se o tempo limite foi atingido, paramos o processamento
        if tempo_limite_obs is not None and tempo_relativo_atual > tempo_limite_obs:
            break

        if (curr_frame - frame_inicial) % intervalo == 0:
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                media_r = np.mean(crop[:, :, 2].astype(np.float64))
                media_g = np.mean(crop[:, :, 1].astype(np.float64))
                media_b = np.mean(crop[:, :, 0].astype(np.float64))
                
                dados_rgb.append([media_r, media_g, media_b])
                tempos.append(tempo_relativo_atual)
        
        if (curr_frame - frame_inicial) % 100 == 0:
             print(f" Processando frame: {curr_frame} (Tempo Obs: {tempo_relativo_atual:.2f}s)", end='\r')
        
        curr_frame += 1
        
    cap.release()

    # --- EQUALIZAÇÃO FINAL ---
    if tempo_limite_obs is not None and len(tempos) > 0 and tempos[-1] < tempo_limite_obs:
        tempos.append(tempo_limite_obs)
        dados_rgb.append(dados_rgb[-1])

    print("\nAnálise concluída.")
    return np.array(tempos), np.array(dados_rgb)

# ==========================================
# PARÂMETROS DE CONTROLE (AJUSTE AQUI)
# ==========================================
caminho_original = r"c:\Users\Micro\Documents\videos_amostras\Primeira_Rodada-20260320T213054Z-3-001\Azul_151535.mp4"
video_file = Path(caminho_original.replace("\\", "/")) #replace para n precisar mudar manualmente o \ da linha de caminho do dwindows para / como um caminho reconhecivel
# Measure-Command { python   analise_do_vd_das_amostras.py }

z_margem_seguranca = 100 
voltar_n_segundos = 10.0 
tempo_de_observacao_final = 60.0 + voltar_n_segundos # Tempo que o gráfico e a linha devem atingir juntos.
threshold_brilho = 180   # Ajustado para 180 para detectar melhor o vermelho
fps_analise = 10         
modo_manual_forcado = False 
conferir_roi_interativamente = True 

# ==========================================
# LÓGICA PRINCIPAL DE EXECUÇÃO
# ==========================================
if video_file.exists():
    try:
        # Primeiro pegamos um frame onde a amostra certamente está acesa (ex: frame 300)
        cap_temp = cv2.VideoCapture(str(video_file))
        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, 300) 
        ret_temp, frame_ref = cap_temp.read()
        cap_temp.release()

        if frame_ref is None:
            raise Exception("Não foi possível ler o vídeo para referência de ROI.")

        # 1. Definimos a ROI primeiro (seja automático ou manual)
        roi_final = None
        if not modo_manual_forcado:
            roi_auto_tentativa = detectar_amostra_automatica(frame_ref, z_margem=z_margem_seguranca)
            if conferir_roi_interativamente:
                roi_final = validar_roi_interativamente(frame_ref, roi_auto_tentativa, threshold_brilho, voltar_n_segundos)
            else:
                roi_final = roi_auto_tentativa
            
        if roi_final is None:
            roi_final = selecionar_roi_manual(video_file, frame_ref)

        # 2. Localizamos a queda real baseada APENAS na ROI selecionada
        idx_queda = localizar_queda_na_roi(video_file, roi_final, threshold_brilho)
        fps = cv2.VideoCapture(str(video_file)).get(cv2.CAP_PROP_FPS)
        idx_start = max(0, idx_queda - int(voltar_n_segundos * fps))

        # 3. Analisamos o vídeo
        if roi_final and roi_final[2] > 0:
            t_relativo, rgb = analisar_video_puro(video_file, roi_final, idx_start, fps_alvo=fps_analise, tempo_limite_obs=tempo_de_observacao_final)
            
            pasta_do_script = Path(os.getcwd()) 
            nome_base = video_file.stem 
            
            # Exportação de Dados para CSV
            df_export = pd.DataFrame({
                'Tempo_s': t_relativo,
                'R_Puro': rgb[:, 0],
                'G_Puro': rgb[:, 1],
                'B_Puro': rgb[:, 2],
                'Media_Total': np.mean(rgb, axis=1)
            })
            
            csv_output = pasta_do_script / f"{nome_base}.csv"
            df_export.to_csv(csv_output, index=False)
            print(f"Dados exportados na pasta do script: {csv_output}")

            # 5. Plotagem Científica (Matplotlib)
            plt.figure(figsize=(12, 6))
            
            plt.plot(t_relativo, rgb[:, 0], 'red', label='Vermelho (Puro)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 1], 'green', label='Verde (Puro)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 2], 'blue', label='Azul (Puro)', linewidth=1.5)
            plt.plot(t_relativo, np.mean(rgb, axis=1), 'black', linestyle='--', label='Média Total', alpha=0.7)
            
            # Linha indicando o momento exato detectado do desligamento
            plt.axvline(x=voltar_n_segundos, color='gray', linestyle=':', label='Desligamento Detectado')

            plt.title(f"Análise do decaimento de Luminescência: {video_file.name}")
            plt.xlabel(f"Tempo de Observação (s)")
            plt.ylabel("Intensidade Média Bruta (0-255)")
            
            plt.xlim(0, tempo_de_observacao_final)
            plt.gca().set_xmargin(0) 

            plt.legend()
            plt.grid(True, which='both', linestyle=':', alpha=0.5)
            
            graph_output = pasta_do_script / f"{nome_base}.png"
            plt.savefig(graph_output, dpi=300)
            print(f"Gráfico salvo em: {graph_output}")
            
            plt.show()
        else:
            print("Análise cancelada: ROI não definido.")

    except Exception as e:
        print(f"Ocorreu um erro crítico: {e}")
        cv2.destroyAllWindows() 

else:
    print(f"Erro: Arquivo não encontrado.")

# O threshold deve ser usado com cuidado pois caso a camera n captar ou a excitação n for suficiente para alcançar o maximo ele nunca chegara aos 235 que é o valor usado
# Por isso posso diminuir e alterar esse valor para se adequar ao nivel de excitação do video
# E caso a captação do video for muito sensivel eu tbm posso aumentalo ja que ele pega pela Intensidade de Brilho Geral (Escala de Cinza) e visualiza com a cor com mais intesidade e checa se ela esta abaixo do threshold
# Measure-Command { python seu_arquivo.py }
