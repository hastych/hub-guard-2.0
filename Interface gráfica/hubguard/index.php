<?php session_start(); ?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HubGuard - Login</title>
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
        .login-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .form-group { margin-bottom: 15px; }
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .feedback {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .error { background: #f8d7da; color: #721c24; }
        .success { background: #d4edda; color: #155724; }
        .info { margin-top: 20px; font-size: 12px; color: #999; }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>🔐 HubGuard</h1>
        <p>Sistema de Gerenciamento de Laboratório</p>
        
        <?php if(isset($_GET['erro'])): ?>
            <div class="feedback error">❌ Usuário ou senha inválidos</div>
        <?php endif; ?>
        
        <?php if(isset($_GET['logout'])): ?>
            <div class="feedback success">✅ Você saiu do sistema</div>
        <?php endif; ?>
        
        <form method="POST" action="login.php">
            <div class="form-group">
                <input type="text" name="usuario" placeholder="Usuário" required autofocus>
            </div>
            <div class="form-group">
                <input type="password" name="senha" placeholder="Senha" required>
            </div>
            <button type="submit">Entrar</button>
        </form>
        
        <div class="info">
            <p>📱 O sistema enviará confirmação por WhatsApp 1h antes da aula</p>
        </div>
    </div>
</body>
</html>