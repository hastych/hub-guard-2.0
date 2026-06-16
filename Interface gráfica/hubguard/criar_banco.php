<?php
echo "<h1>Configurando Banco de Dados...</h1>";

try {
    $pdo = new PDO('sqlite:' . __DIR__ . '/hubguard.db');
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    
    // Tabela usuarios
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            usuario TEXT UNIQUE NOT NULL,
            senha TEXT NOT NULL,
            telefone TEXT,
            tipo TEXT DEFAULT 'user',
            ativo INTEGER DEFAULT 1
        )
    ");
    echo "✅ Tabela usuarios criada!<br>";
    
    // Admin
    $check = $pdo->query("SELECT * FROM usuarios WHERE usuario = 'LAB_Redes'");
    if (!$check->fetch()) {
        $stmt = $pdo->prepare("INSERT INTO usuarios (nome, usuario, senha, telefone, tipo) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute(['Administrador Redes', 'LAB_Redes', password_hash('LAB@2013', PASSWORD_DEFAULT), '11999999999', 'admin']);
        echo "✅ Admin criado!<br>";
    }
    
    // Usuário teste
    $check = $pdo->query("SELECT * FROM usuarios WHERE usuario = 'aluno'");
    if (!$check->fetch()) {
        $stmt = $pdo->prepare("INSERT INTO usuarios (nome, usuario, senha, telefone, tipo) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute(['Aluno Teste', 'aluno', password_hash('aluno123', PASSWORD_DEFAULT), '11988888888', 'user']);
        echo "✅ Usuário teste criado!<br>";
    }
    
    // Tabela horarios
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS horarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dia_semana TEXT,
            materia TEXT,
            professor TEXT,
            horario_inicio TEXT,
            horario_fim TEXT,
            sala TEXT DEFAULT 'Lab 1'
        )
    ");
    echo "✅ Tabela horarios criada!<br>";
    
    echo "<br><strong style='color:green'>🎉 SISTEMA CONFIGURADO!</strong><br>";
    echo "<a href='index.php'>🔐 Ir para o login →</a>";
    
} catch (PDOException $e) {
    echo "Erro: " . $e->getMessage();
}
?>