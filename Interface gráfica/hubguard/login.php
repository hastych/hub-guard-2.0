<?php
session_start();
require_once 'config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: index.php');
    exit;
}

$usuario = trim($_POST['usuario'] ?? '');
$senha = $_POST['senha'] ?? '';

// ADMIN FIXO
if ($usuario === ADMIN_USER && $senha === ADMIN_PASS) {
    $_SESSION['user_id'] = 1;
    $_SESSION['user_nome'] = 'Administrador Redes';
    $_SESSION['user_usuario'] = ADMIN_USER;
    $_SESSION['user_tipo'] = 'admin';
    header('Location: menu.php');
    exit;
}

// Verificar no banco SQLite
try {
    $pdo = getConnection();
    $stmt = $pdo->prepare("SELECT * FROM usuarios WHERE usuario = ? AND ativo = 1");
    $stmt->execute([$usuario]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($user && password_verify($senha, $user['senha'])) {
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['user_nome'] = $user['nome'];
        $_SESSION['user_usuario'] = $user['usuario'];
        $_SESSION['user_tipo'] = $user['tipo'];
        $_SESSION['user_telefone'] = $user['telefone'];
        header('Location: menu.php');
    } else {
        header('Location: index.php?erro=1');
    }
} catch (PDOException $e) {
    header('Location: index.php?erro=1');
}
?>