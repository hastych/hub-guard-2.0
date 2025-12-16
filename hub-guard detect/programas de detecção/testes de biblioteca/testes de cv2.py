import cv2

# Caminho da imagem
caminho_imagem = r'C:\Users\Pichau\Desktop\testes hub-guard\hub-guard detect\testes de biblioteca\imagem para testes\pessoa 1.jpg'

# Carrega a imagem
image = cv2.imread(caminho_imagem)

# Verifica se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Exibe a imagem em uma janela
    cv2.imshow('Imagem', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()