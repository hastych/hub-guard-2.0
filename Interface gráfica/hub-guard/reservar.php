<?php
include __DIR__ . '/conexao.php';

$lab = $_POST['lab'] ?? '';
$usuario = $_POST['usuario'] ?? '';
$data_hora = $_POST['data_hora'] ?? '';

if (!empty($lab) && !empty($usuario) && !empty($data_hora)) {
    try {
        $stmt = $conn->prepare("
            INSERT INTO reservas (laboratorio, usuario, data_hora) 
            VALUES (:lab, :usuario, :data_hora)
        ");
        $stmt->bindParam(':lab', $lab);
        $stmt->bindParam(':usuario', $usuario);
        $stmt->bindParam(':data_hora', $data_hora);
        $stmt->execute();

        echo "<script>alert('Reserva realizada com sucesso!'); window.location.href='menu.html';</script>";
    } catch (PDOException $e) {
        die("Erro ao registrar reserva: " . $e->getMessage());
    }
} else {
    echo "<script>alert('Por favor, preencha todos os campos!'); window.location.href='reserve.html';</script>";
}
?>