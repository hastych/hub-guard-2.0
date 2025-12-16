<?php
include __DIR__ . '/conexao.php';

$username = $_POST['username'] ?? '';
$password = $_POST['password'] ?? '';

if (!empty($username) && !empty($password)) {
    $stmt = $conn->prepare("SELECT * FROM usuarios WHERE username = :username AND password = :password");
    $stmt->bindParam(':username', $username);
    $stmt->bindParam(':password', $password);
    $stmt->execute();

    if ($stmt->rowCount() > 0) {
        header("Location: menu.html");
        exit();
    } else {
        echo "<script>alert('Usuário ou senha inválidos!'); window.location.href='index.html';</script>";
    }
} else {
    echo "<script>alert('Por favor, preencha todos os campos!'); window.location.href='index.html';</script>";
}
?>