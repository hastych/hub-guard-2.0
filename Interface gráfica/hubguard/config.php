<?php
// SQLite - NÃO USA MYSQL!
define('ADMIN_USER', 'LAB_Redes');
define('ADMIN_PASS', 'LAB@2013');

function getConnection() {
    try {
        $pdo = new PDO('sqlite:' . __DIR__ . '/hubguard.db');
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $pdo;
    } catch (PDOException $e) {
        die("Erro de conexão: " . $e->getMessage());
    }
}
?>