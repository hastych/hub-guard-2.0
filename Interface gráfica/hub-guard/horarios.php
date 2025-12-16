<?php
include __DIR__ . '/conexao.php';

$query = "SELECT * FROM reservas ORDER BY data_hora";
$stmt = $conn->query($query);

echo "<!DOCTYPE html>
<html lang='pt-br'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Hub Guard - Horários</title>
    <link rel='stylesheet' href='style.css'>
</head>
<body>
    <header>
        <h1>Horários dos Laboratórios</h1>
    </header>
    <div class='container'>
        <table>
            <thead>
                <tr>
                    <th>Laboratório</th>
                    <th>Usuário</th>
                    <th>Data e Hora</th>
                </tr>
            </thead>
            <tbody>";
while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
    echo "<tr>
            <td>{$row['laboratorio']}</td>
            <td>{$row['usuario']}</td>
            <td>{$row['data_hora']}</td>
          </tr>";
}
echo "      </tbody>
        </table>
    </div>
</body>
</html>";
?>