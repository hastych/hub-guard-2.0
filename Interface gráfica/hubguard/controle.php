<?php
session_start();

if (!isset($_SESSION['user_id'])) {
    header('Location: index.php');
    exit;
}

$mensagem = '';
$tipo_mensagem = '';

// ===== IP DO ESP32 =====
$esp_ip = '192.168.0.200';

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['acao'])) {
    $acao = $_POST['acao'];
    
    if ($acao === 'abrir' || $acao === 'fechar') {
        
        $url = "http://{$esp_ip}/control?acao=" . $acao;
        
        $options = [
            'http' => [
                'method' => 'GET',
                'timeout' => 5,
                'ignore_errors' => true
            ]
        ];
        
        $context = stream_context_create($options);
        $response = @file_get_contents($url, false, $context);
        
        if ($response !== false) {
            $dados = json_decode($response, true);
            if ($dados && isset($dados['status']) && $dados['status'] === 'ok') {
                $mensagem = ($acao === 'abrir') ? '✅ Porta destrancada!' : '✅ Porta trancada!';
                $tipo_mensagem = 'success';
            } else {
                $mensagem = '⚠️ Comando recebido, mas resposta inesperada do ESP';
                $tipo_mensagem = 'error';
            }
        } else {
            $mensagem = '⚠️ ESP32 não respondeu. Verifique o IP e a conexão.';
            $tipo_mensagem = 'error';
        }
    }
}

// Buscar status atual
$status_atual = 'Desconhecido';
$url_status = "http://{$esp_ip}/status";
$status_response = @file_get_contents($url_status);
if ($status_response !== false) {
    $status_data = json_decode($status_response, true);
    if ($status_data && isset($status_data['status'])) {
        $status_atual = $status_data['status'] === 'aberto' ? 'ABERTO 🔓' : 'FECHADO 🔒';
    }
}
?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Controle - HubGuard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .icone-fechadura { font-size: 80px; margin: 20px 0; }
        .status-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            font-size: 18px;
        }
        .status-aberto { color: #28a745; font-weight: bold; }
        .status-fechado { color: #dc3545; font-weight: bold; }
        .btn {
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            color: white;
            margin: 5px;
        }
        .btn-abrir { background: #28a745; }
        .btn-fechar { background: #dc3545; }
        .feedback {
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .ip-info { font-size: 12px; color: #999; margin: 10px 0; }
        .btn-voltar {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
        }
        .info-luz {
            margin-top: 30px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 15px;
        }
        .em-breve { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🔓 Controle da Fechadura</h1>
        
        <?php if ($mensagem): ?>
        <div class="feedback <?php echo $tipo_mensagem; ?>"><?php echo $mensagem; ?></div>
        <?php endif; ?>
        
        <div class="icone-fechadura"><?php echo $status_atual === 'ABERTO 🔓' ? '🔓' : '🔒'; ?></div>
        
        <div class="status-box">
            Status: <span class="<?php echo $status_atual === 'ABERTO 🔓' ? 'status-aberto' : 'status-fechado'; ?>">
                <?php echo $status_atual; ?>
            </span>
        </div>
        
        <div>
            <form method="POST" style="display:inline;">
                <button type="submit" name="acao" value="abrir" class="btn btn-abrir">🚪 Destrancar</button>
            </form>
            <form method="POST" style="display:inline;">
                <button type="submit" name="acao" value="fechar" class="btn btn-fechar">🔒 Trancar</button>
            </form>
        </div>
        
        <div class="ip-info">📡 ESP32: <?php echo $esp_ip; ?></div>
        
        <div class="info-luz">
            <h3>💡 Controle de Luz</h3>
            <div class="em-breve">🚧 Em breve</div>
        </div>
        
        <button onclick="location.href='menu.php'" class="btn-voltar">← Voltar</button>
    </div>
</body>
</html>