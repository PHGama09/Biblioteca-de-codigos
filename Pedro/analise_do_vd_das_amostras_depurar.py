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

def encontrar_frame_ideal_por_tempo(video_path, threshold_queda=235, voltar_n_segundos=1.0):
    """
    Analisa o vídeo para encontrar o momento em que a luz apaga e retrocede 
    N segundos para encontrar o frame ideal de detecção da amostra.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        print("Erro: Não foi possível obter o FPS do vídeo.")
        return None, 0
    
    frame_idx = 0
    ponto_queda_frame = 0
    encontrou_excitacao = False
    
    print(f"Buscando evento de queda de intensidade (Threshold: {threshold_queda})...")
    
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
    msg_titulo = f"Conferencia ROI - Frame Ideal (T-{voltar_n_segundos}s)"
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
    print("Verifique a janela na tela:")
    print(" - Pressione ENTER (ou Espaço) para ACEITAR o ROI automático (Verde).")
    print(" - Pressione 'M' (ou 'm') para REJEITAR e definir manualmente.")
    print(" - Pressione ESC para cancelar o programa.")
    
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
    
    # Pega o total de frames apenas para o print de progresso
    total_frames_analise = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - frame_inicial
    
    print(f"\nIniciando análise RGB pura a partir do frame {frame_inicial}...")
    print(f"Área analisada (ROI): x={x}, y={y}, Tamanho={w}x{h} pixels.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Processa apenas nos frames de intervalo para atingir o fps_alvo
        if (curr_frame - frame_inicial) % intervalo == 0:
            # Corta a região (Crop) [y:y+h, x:x+w] no formato NumPy
            crop = frame[y:y+h, x:x+w]
            
            if crop.size > 0:
                # DADOS PUROS: Extração direta canal por canal usando float64.
                # Isso garante precisão científica e impede erros de arredondamento
                # ou estouro de memória que contaminariam a média.
                # OpenCV usa BGR, invertemos para RGB puro.
                media_r = np.mean(crop[:, :, 2].astype(np.float64))
                media_g = np.mean(crop[:, :, 1].astype(np.float64))
                media_b = np.mean(crop[:, :, 0].astype(np.float64))
                
                dados_rgb.append([media_r, media_g, media_b])
                tempos.append(curr_frame / fps_orig)
        
        # Print simples de progresso (a cada 100 frames)
        if (curr_frame - frame_inicial) % 100 == 0:
             print(f" Processando frame: {curr_frame} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", end='\r')
        
        curr_frame += 1
        
    cap.release()
    print("\nAnálise concluída.")
    return np.array(tempos), np.array(dados_rgb)

# ==========================================
# PARÂMETROS DE CONTROLE (AJUSTE AQUI)
# ==========================================
# Use barras normais '/' no caminho do arquivo, mesmo no Windows.
video_file = Path("c:/Users/Micro/Documents/videos_amostras/Vermelho_154017.mp4")

z_margem_seguranca = 40  # Margem para dentro da resina (pixels). Mantenha alto para pegar o centro estável.
voltar_n_segundos = 15.0 # Segundos ANTES da luz UV apagar para iniciar o gráfico.
threshold_brilho = 235   # Sensibilidade (0-255) do pixel mais brilhante para detectar o desligamento da UV.
fps_analise = 5          # Resolução temporal do gráfico (pontos por segundo).
modo_manual_forcado = False # Mude para True se você NUNCA quiser usar o automático.
conferir_roi_interativamente = False # Nova variável: Se False, pula a janela verde e vai direto (ou manual se falhar).

# ==========================================
# LÓGICA PRINCIPAL DE EXECUÇÃO
# ==========================================
if video_file.exists():
    try:
        # 1. Busca o momento exato de desligamento e retrocede o tempo para o frame ideal
        frame_ref, idx_start = encontrar_frame_ideal_por_tempo(video_file, threshold_brilho, voltar_n_segundos)
        
        if frame_ref is None:
            raise Exception("Não foi possível ler o vídeo para detecção de ROI.")

        # 2. Define a ROI (Automática ou Manual de Backup)
        roi_final = None
        
        # Tenta o automático primeiro (se não estiver forçado o manual)
        if not modo_manual_forcado:
            roi_auto_tentativa = detectar_amostra_automatica(frame_ref, z_margem=z_margem_seguranca)
            
            # Validação Interativa: Mostra na tela para o usuário conferir ANTES de prosseguir
            if conferir_roi_interativamente:
                roi_final = validar_roi_interativamente(frame_ref, roi_auto_tentativa, threshold_brilho, voltar_n_segundos)
            else:
                # Se não for para conferir, assume o automático (que pode ser None se falhar)
                roi_final = roi_auto_tentativa
            
        # Fallback Manual: Se forçado manual OU se o automático falhou internamente 
        # OU se o usuário rejeitou o automático na validação interativa.
        if roi_final is None:
            roi_final = selecionar_roi_manual(video_file, frame_ref)

        # 3. Processamento Final: Analisa os dados puros a partir do ponto de interesse definido
        if roi_final and roi_final[2] > 0:
            # t: vetor de tempo; rgb: matriz Nx3 [R, G, B]
            t, rgb = analisar_video_puro(video_file, roi_final, idx_start, fps_alvo=fps_analise)
            
            # --- MUDANÇA AQUI: DEFININDO PASTA DO SCRIPT ---
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
            
            # Define o salvamento na pasta do script
            csv_output = pasta_do_script / f"{nome_base}.csv"
            df_export.to_csv(csv_output, index=False)
            print(f"Dados exportados na pasta do script: {csv_output}")

            # 5. Plotagem Científica (Matplotlib)
            plt.figure(figsize=(12, 6))
            
            # Faz o gráfico começar do 'tempo zero' da análise (ponto definido por voltar_n_segundos)
            plt.plot(t_relativo, rgb[:, 0], 'red', label='Vermelho (Puro)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 1], 'green', label='Verde (Puro)', linewidth=1.5)
            plt.plot(t_relativo, rgb[:, 2], 'blue', label='Azul (Puro)', linewidth=1.5)
            
            # Linha de Intensidade Média Total (Preto tracejado, média aritmética simples dos 3)
            plt.plot(t_relativo, np.mean(rgb, axis=1), 'black', linestyle='--', label='Média Total', alpha=0.7)

            # plt.title(f"Cinética de Luminescência: {video_file.name}")
            # plt.title(f"Cinética de Luminescência Bruta: {video_file.name}\n(ROI: {roi_final[2]}x{roi_final[3]} pixels, Z={z_margem_seguranca})")
            plt.title(f"Cinética de Luminescência: {video_file.name}")
            plt.xlabel(f"Tempo de Observação (s)")
            plt.ylabel("Intensidade Média Bruta (0-255)")
            plt.legend()
            # Grade suave para facilitar a leitura
            plt.grid(True, which='both', linestyle=':', alpha=0.5)
            
            # Salva a imagem do gráfico antes de exibir na pasta do script
            graph_output = pasta_do_script / f"{nome_base}.png"
            plt.savefig(graph_output, dpi=300)
            print(f"Gráfico salvo em: {graph_output}")
            
            plt.show()
        else:
            print("Análise cancelada: ROI não definido.")

    except Exception as e:
        print(f"Ocorreu um erro crítico durante o processamento: {e}")
        cv2.destroyAllWindows() # Garante que as janelas do OpenCV fechem em caso de erro

else:
    print(f"Erro: O arquivo de vídeo não foi encontrado no caminho especificado:\n{video_file}")


# O threshold deve ser usado com cuidado pois caso a camera n captar ou a excitação n for suficiente para alcançar o maximo ele nunca chegara aos 235 que é o valor usado
# Por isso posso diminuir e alterar esse valor para se adequar ao nivel de excitação do video
# E caso a captação do video for muito sensivel eu tbm posso aumentalo ja que ele pega pela Intensidade de Brilho Geral (Escala de Cinza) e visualiza com a cor com mais intesidade e checa se ela esta abaixo do threshold
