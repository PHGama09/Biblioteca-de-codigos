import cv2 # Importa a biblioteca OpenCV, que aqui será usada apenas para "tocar" o vídeo e ler suas propriedades internas.
import matplotlib.pyplot as plt # Importa a biblioteca Pyplot do Matplotlib. É o padrão em Python para desenhar gráficos científicos 2D.
from collections import Counter # Importa uma ferramenta nativa do Python projetada especificamente para contar coisas rapidamente e agrupar resultados.

# Cria a função principal. Tudo o que estiver recuado (identado) abaixo faz parte dessa função.
# Ela recebe como parâmetro obrigatório a variável 'caminho_video' (onde o vídeo está no PC).
def plotar_fps_do_video(caminho_video):
      
    # O comando VideoCapture, em vez de receber um número (0, 1) para abrir uma câmera, 
    # recebe o caminho de um arquivo para abrir e ler o vídeo direto do disco rígido.
    cap = cv2.VideoCapture(caminho_video)
    
    # Trava de segurança: isOpened() verifica se o arquivo realmente existe ou se o caminho está certo.
    # Se der falso (not), ele imprime o erro e usa o 'return' para abortar a função imediatamente.
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo. Verifique o caminho.")
        return

    # Inicia o nosso contador. Ele funciona como um dicionário. 
    # Vai guardar a informação no formato: {Segundo_0: 30 frames, Segundo_1: 28 frames, Segundo_2: 30 frames...}
    contagem_fps = Counter()

    # Imprime uma mensagem para você saber que o programa não travou, pois processar vídeos longos demora um pouco.
    print("Processando o vídeo... Isso pode levar alguns segundos.")
    
    # Inicia o loop infinito para ler o vídeo frame a frame, do início ao fim.
    while True:
        # Puxa o próximo quadro do arquivo. 
        # 'ret' vira False quando o vídeo acaba (não tem mais quadros para ler).
        ret, frame = cap.read()
        
        # Se 'ret' for Falso (fim do vídeo), ele quebra o loop e vai para a criação do gráfico.
        if not ret:
            break 

        # cap.get acessa os metadados daquele quadro específico. 
        # CAP_PROP_POS_MSEC pergunta ao arquivo: "Em qual milissegundo exato do vídeo este quadro deveria aparecer?"
        tempo_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Converte milissegundos para segundos. Dividimos por 1000 e usamos int() para arredondar para baixo.
        # Exemplo: Se o quadro está em 1450ms (1.45s), o int() corta os decimais e transforma no "Segundo 1".
        segundo_atual = int(tempo_ms / 1000.0)
        
        # Aqui é onde a mágica acontece. Para cada quadro que passa no loop, 
        # ele vai no 'segundo_atual' (ex: Segundo 1) e soma +1 na conta de frames daquele segundo.
        contagem_fps[segundo_atual] += 1

    # Quando o loop acaba, fechamos o arquivo de vídeo para liberar a memória RAM do computador.
    cap.release()

    # --- INÍCIO DA PREPARAÇÃO DOS DADOS PARA O GRÁFICO ---
    
    # Pega todas as "chaves" do contador (os segundos: 0, 1, 2, 3...) e os ordena do menor para o maior (ordem cronológica).
    segundos = sorted(contagem_fps.keys())
    
    # Cria uma lista pegando a quantidade de frames que contamos para cada um daqueles segundos ordenados na linha acima.
    fps = [contagem_fps[sec] for sec in segundos]

    # --- INÍCIO DA DESENHO DO GRÁFICO (MATPLOTLIB) ---
    
    # Cria uma tela de pintura em branco (figura) com dimensões de 10 polegadas de largura por 5 de altura.
    plt.figure(figsize=(10, 5))
    
    # Plota a linha! Eixo X = Segundos, Eixo Y = FPS. 
    # marker='o' coloca bolinhas nos pontos exatos; linestyle='-' liga os pontos com uma linha; color='#1f77b4' é um tom de azul.
    plt.plot(segundos, fps, marker='o', linestyle='-', color='#1f77b4')
    
    # Adiciona o título no topo do gráfico.
    plt.title('Variação de FPS por Segundo')
    
    # Dá um nome ao eixo horizontal (X).
    plt.xlabel('Tempo da reprodução (Segundos)')
    
    # Dá um nome ao eixo vertical (Y).
    plt.ylabel('Quantidade de Frames (FPS)')
    
    # Liga a grade de fundo do gráfico (grid). Usa linha tracejada ('--') com transparência de 70% (alpha=0.7) para não poluir.
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Define os limites do eixo Y. Começa no 0. 
    # O limite superior será o "Valor máximo de FPS encontrado + 10" para dar uma margem de respiro no topo. 
    # Se o vídeo estiver vazio e der erro de FPS nulo, ele usa o valor fixo de 60 por segurança.
    plt.ylim(0, max(fps) + 10 if fps else 60) 
    
    # Função que ajusta automaticamente os espaçamentos das bordas para o texto não ficar cortado fora da janela.
    plt.tight_layout()
    
    # Exibe a janela interativa do gráfico na sua tela do computador (onde você pode dar zoom ou salvar a imagem).
    plt.show()

# ==========================================
# ÁREA DE EXECUÇÃO DO SCRIPT
# ==========================================

# Cria uma string (texto) contendo o caminho exato onde o vídeo está salvo no seu HD.
caminho_do_seu_video = "C:/Users/imene/Downloads/Projeto_Luminescencia/WIN_20260517_18_22_37_Pro.mp4"

# Chama a função que criamos lá em cima, "injetando" o caminho do vídeo dentro dela para que ela inicie todo o trabalho.
plotar_fps_do_video(caminho_do_seu_video)
