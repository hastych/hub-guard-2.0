<?php
session_start();

if (!isset($_SESSION['user_id'])) {
    header('Location: index.php');
    exit;
}

$userName = $_SESSION['user_nome'];
$isAdmin = ($_SESSION['user_tipo'] === 'admin');
?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HubGuard - Menu</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .menu-header { text-align: center; color: white; margin-bottom: 40px; }
        .menu-header h1 { font-size: 2rem; margin-bottom: 10px; }
        .menu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .menu-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .menu-card:hover:not(.disabled) {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .icon { font-size: 48px; margin-bottom: 10px; }
        .menu-card h3 { margin-bottom: 10px; color: #333; }
        .menu-card p { color: #666; font-size: 14px; }
        .menu-card.disabled { opacity: 0.6; cursor: not-allowed; }
        .menu-card.admin {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .menu-card.admin h3, .menu-card.admin p { color: white; }
        .menu-card.logout { background: #f8d7da; }
        .breve { color: #ff9800; font-weight: bold; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="menu-header">
            <h1>Olá, <?php echo htmlspecialchars($userName); ?>! 👋</h1>
            <p><?php echo date('d/m/Y'); ?></p>
        </div>
        
        <div class="menu-grid">
            <div class="menu-card" onclick="location.href='horarios.php'">
                <div class="icon">📅</div>
                <h3>Ver Horários</h3>
                <p>Grade completa das aulas</p>
            </div>
            
            <?php if ($isAdmin): ?>
            <div class="menu-card admin" onclick="location.href='admin_horarios.php'">
                <div class="icon">📚</div>
                <h3>Gerenciar Horários</h3>
                <p>Adicionar/editar/excluir aulas</p>
            </div>
            <?php endif; ?>
            
            <div class="menu-card" onclick="location.href='controle.php'">
                <div class="icon">🔓</div>
                <h3>Abrir Porta</h3>
                <p>Controlar a fechadura do laboratório</p>
            </div>
            
            <div class="menu-card disabled">
                <div class="icon">💡</div>
                <h3>Controlar Luz</h3>
                <p class="breve">🚧 Em breve</p>
            </div>
            
            <?php if ($isAdmin): ?>
            <div class="menu-card admin" onclick="location.href='admin.php'">
                <div class="icon">👑</div>
                <h3>Gerenciar Usuários</h3>
                <p>Cadastrar e gerenciar pessoas</p>
            </div>
            <?php endif; ?>
            
            <div class="menu-card logout" onclick="location.href='logout.php'">
                <div class="icon">🚪</div>
                <h3>Sair</h3>
                <p>Encerrar sessão</p>
            </div>
        </div>
    </div>
</body>
</html>