<?php
session_start();

if (!isset($_SESSION['user_id'])) {
    http_response_code(403);
    exit;
}

$esp_ip = '192.168.0.200';

$url = "http://{$esp_ip}/status";
$response = @file_get_contents($url);

if ($response !== false) {
    $dados = json_decode($response, true);
    if ($dados && isset($dados['status'])) {
        echo json_encode(['status' => $dados['status']]);
        exit;
    }
}

echo json_encode(['status' => 'desconhecido']);
?>