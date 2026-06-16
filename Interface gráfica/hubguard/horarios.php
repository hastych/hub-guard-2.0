<?php
session_start();

if (!isset($_SESSION['user_id'])) {
    header('Location: index.php');
    exit;
}

require_once 'config.php';
$pdo = getConnection();

$dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta'];
$horarios_por_dia = [];

foreach ($dias as $dia) {
    $stmt = $pdo->prepare("SELECT * FROM horarios WHERE dia_semana = ? ORDER BY horario_inicio");
    $stmt->execute([$dia]);
    $horarios_por_dia[$dia] = $stmt->fetchAll();
}
?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Horários - HubGuard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .grade-horarios {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
            margin: 30px 0;
        }
        .dia-coluna {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
        }
        .dia-coluna h3 {
            color: #667eea;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .aula-card {
            background: white;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .aula-materia { font-weight: bold; color: #333; }
        .aula-professor { font-size: 12px; color: #666; margin: 5px 0; }
        .aula-horario { font-size: 11px; color: #999; }
        .aula-sala { font-size: 10px; color: #667eea; margin-top: 3px; }
        .vazio { text-align: center; color: #999; padding: 20px; }
        .btn-voltar {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
        }
        .legenda { font-size: 12px; color: #999; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📅 Horário das Aulas</h1>
            <p>Laboratório de Informática</p>
            
            <div class="grade-horarios">
                <?php foreach ($dias as $dia): ?>
                <div class="dia-coluna">
                    <h3><?php echo $dia; ?></h3>
                    <?php if (empty($horarios_por_dia[$dia])): ?>
                        <div class="vazio">📭 Sem aulas</div>
                    <?php else: ?>
                        <?php foreach ($horarios_por_dia[$dia] as $aula): ?>
                        <div class="aula-card">
                            <div class="aula-materia"><?php echo htmlspecialchars($aula['materia']); ?></div>
                            <div class="aula-professor">👨‍🏫 <?php echo htmlspecialchars($aula['professor']); ?></div>
                            <div class="aula-horario">⏰ <?php echo substr($aula['horario_inicio'], 0, 5); ?> - <?php echo substr($aula['horario_fim'], 0, 5); ?></div>
                            <div class="aula-sala">📍 <?php echo htmlspecialchars($aula['sala']); ?></div>
                        </div>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </div>
                <?php endforeach; ?>
            </div>
            
            <button onclick="location.href='menu.php'" class="btn-voltar">← Voltar ao Menu</button>
            <div class="legenda">💡 Os horários são gerenciados pelo administrador</div>
        </div>
    </div>
</body>
</html>