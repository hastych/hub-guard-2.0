<?php
session_start();

if (!isset($_SESSION['user_id']) || $_SESSION['user_tipo'] !== 'admin') {
    header('Location: index.php');
    exit;
}

require_once 'config.php';
$pdo = getConnection();
$mensagem = '';
$tipo_msg = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['cadastrar'])) {
        $nome = $_POST['nome'];
        $usuario = $_POST['usuario'];
        $senha = password_hash($_POST['senha'], PASSWORD_DEFAULT);
        $telefone = $_POST['telefone'];
        $tipo = $_POST['tipo'];
        
        try {
            $stmt = $pdo->prepare("INSERT INTO usuarios (nome, usuario, senha, telefone, tipo) VALUES (?, ?, ?, ?, ?)");
            $stmt->execute([$nome, $usuario, $senha, $telefone, $tipo]);
            $mensagem = "✅ Usuário cadastrado com sucesso!";
            $tipo_msg = 'success';
        } catch (PDOException $e) {
            $mensagem = "❌ Erro: Usuário já existe!";
            $tipo_msg = 'error';
        }
    }
    
    if (isset($_POST['excluir'])) {
        $id = $_POST['id'];
        $stmt = $pdo->prepare("DELETE FROM usuarios WHERE id = ? AND usuario != 'LAB_Redes'");
        $stmt->execute([$id]);
        $mensagem = "✅ Usuário removido!";
        $tipo_msg = 'success';
    }
}

$usuarios = $pdo->query("SELECT * FROM usuarios ORDER BY id DESC")->fetchAll();
?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Admin - HubGuard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        h2 { margin: 20px 0 10px 0; color: #333; }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background: #f8f9fa; }
        .btn-excluir {
            background: #dc3545;
            padding: 5px 10px;
        }
        .feedback {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .btn-voltar {
            background: #6c757d;
            margin-top: 20px;
        }
        .voce { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>👑 Painel Administrativo</h1>
            <p>Bem-vindo, <?php echo $_SESSION['user_nome']; ?>!</p>
            
            <?php if ($mensagem): ?>
            <div class="feedback <?php echo $tipo_msg; ?>"><?php echo $mensagem; ?></div>
            <?php endif; ?>
            
            <h2>📝 Cadastrar Usuário</h2>
            <form method="POST">
                <div class="form-row">
                    <input type="text" name="nome" placeholder="Nome completo" required>
                    <input type="text" name="usuario" placeholder="Usuário (login)" required>
                </div>
                <div class="form-row">
                    <input type="password" name="senha" placeholder="Senha" required>
                    <input type="tel" name="telefone" placeholder="Telefone (WhatsApp)" required>
                </div>
                <div class="form-row">
                    <select name="tipo">
                        <option value="user">👤 Usuário comum</option>
                        <option value="admin">👑 Administrador</option>
                    </select>
                </div>
                <button type="submit" name="cadastrar">➕ Cadastrar</button>
            </form>
            
            <h2>📋 Usuários Cadastrados</h2>
            <table>
                <thead>
                    <tr><th>Nome</th><th>Usuário</th><th>Telefone</th><th>Tipo</th><th>Ações</th></tr>
                </thead>
                <tbody>
                    <?php foreach ($usuarios as $user): ?>
                    <tr>
                        <td><?php echo htmlspecialchars($user['nome']); ?></td>
                        <td><?php echo htmlspecialchars($user['usuario']); ?></td>
                        <td><?php echo htmlspecialchars($user['telefone']); ?></td>
                        <td><?php echo $user['tipo'] === 'admin' ? '👑 Admin' : '👤 User'; ?></td>
                        <td>
                            <?php if ($user['usuario'] !== 'LAB_Redes'): ?>
                            <form method="POST" style="display:inline">
                                <input type="hidden" name="id" value="<?php echo $user['id']; ?>">
                                <button type="submit" name="excluir" class="btn-excluir" onclick="return confirm('Excluir?')">🗑️</button>
                            </form>
                            <?php else: ?>
                            <span class="voce">(Admin fixo)</span>
                            <?php endif; ?>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
            
            <button onclick="location.href='menu.php'" class="btn-voltar">← Voltar</button>
        </div>
    </div>
</body>
</html>