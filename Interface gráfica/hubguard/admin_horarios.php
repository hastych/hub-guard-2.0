<?php
session_start();

// Só admin entra aqui
if (!isset($_SESSION['user_id']) || $_SESSION['user_tipo'] !== 'admin') {
    header('Location: index.php');
    exit;
}

require_once 'config.php';
$pdo = getConnection();

$mensagem = '';
$tipo_msg = '';

// Processar formulários
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    
    // ADICIONAR HORÁRIO
    if (isset($_POST['adicionar'])) {
        $dia = $_POST['dia'];
        $materia = $_POST['materia'];
        $professor = $_POST['professor'];
        $inicio = $_POST['horario_inicio'];
        $fim = $_POST['horario_fim'];
        $sala = $_POST['sala'] ?? 'Lab 1';
        
        try {
            $stmt = $pdo->prepare("INSERT INTO horarios (dia_semana, materia, professor, horario_inicio, horario_fim, sala) VALUES (?, ?, ?, ?, ?, ?)");
            $stmt->execute([$dia, $materia, $professor, $inicio, $fim, $sala]);
            $mensagem = "✅ Horário adicionado com sucesso!";
            $tipo_msg = 'success';
        } catch (PDOException $e) {
            $mensagem = "❌ Erro: " . $e->getMessage();
            $tipo_msg = 'error';
        }
    }
    
    // EDITAR HORÁRIO
    if (isset($_POST['editar'])) {
        $id = $_POST['id'];
        $dia = $_POST['dia'];
        $materia = $_POST['materia'];
        $professor = $_POST['professor'];
        $inicio = $_POST['horario_inicio'];
        $fim = $_POST['horario_fim'];
        $sala = $_POST['sala'] ?? 'Lab 1';
        
        try {
            $stmt = $pdo->prepare("UPDATE horarios SET dia_semana = ?, materia = ?, professor = ?, horario_inicio = ?, horario_fim = ?, sala = ? WHERE id = ?");
            $stmt->execute([$dia, $materia, $professor, $inicio, $fim, $sala, $id]);
            $mensagem = "✅ Horário atualizado!";
            $tipo_msg = 'success';
        } catch (PDOException $e) {
            $mensagem = "❌ Erro: " . $e->getMessage();
            $tipo_msg = 'error';
        }
    }
    
    // EXCLUIR HORÁRIO
    if (isset($_POST['excluir'])) {
        $id = $_POST['id'];
        try {
            $stmt = $pdo->prepare("DELETE FROM horarios WHERE id = ?");
            $stmt->execute([$id]);
            $mensagem = "✅ Horário removido!";
            $tipo_msg = 'success';
        } catch (PDOException $e) {
            $mensagem = "❌ Erro: " . $e->getMessage();
            $tipo_msg = 'error';
        }
    }
}

// Buscar todos os horários
$horarios = $pdo->query("SELECT * FROM horarios ORDER BY dia_semana, horario_inicio")->fetchAll();
$dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta'];
?>
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gerenciar Horários - HubGuard</title>
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
            margin-bottom: 20px;
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        h2 { margin: 20px 0 10px 0; color: #333; }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        .form-row2 {
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
        .btn-excluir { background: #dc3545; padding: 5px 10px; }
        .btn-editar { background: #ffc107; color: #333; padding: 5px 10px; }
        .btn-salvar { background: #28a745; padding: 5px 10px; }
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
        .btn-adicionar { background: #28a745; }
        .btn-cancelar { background: #dc3545; }
        .input-edit { padding: 5px; border: 1px solid #ddd; border-radius: 4px; width: 100%; }
        .select-edit { padding: 5px; border: 1px solid #ddd; border-radius: 4px; width: 100%; }
        .acoes { display: flex; gap: 5px; flex-wrap: wrap; }
        .edit-row { background: #fff3cd; }
        .edit-row td { padding: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📚 Gerenciar Horários</h1>
            <p>Bem-vindo, <?php echo $_SESSION['user_nome']; ?>! Gerencie os horários das aulas.</p>
            
            <?php if ($mensagem): ?>
            <div class="feedback <?php echo $tipo_msg; ?>"><?php echo $mensagem; ?></div>
            <?php endif; ?>
            
            <!-- Formulário para ADICIONAR horário -->
            <h2>➕ Adicionar Novo Horário</h2>
            <form method="POST">
                <div class="form-row">
                    <select name="dia" required>
                        <option value="">Dia da semana</option>
                        <?php foreach ($dias as $d): ?>
                        <option value="<?php echo $d; ?>"><?php echo $d; ?></option>
                        <?php endforeach; ?>
                    </select>
                    <input type="text" name="materia" placeholder="Matéria" required>
                    <input type="text" name="professor" placeholder="Professor" required>
                </div>
                <div class="form-row2">
                    <input type="time" name="horario_inicio" required>
                    <input type="time" name="horario_fim" required>
                </div>
                <div class="form-row2">
                    <input type="text" name="sala" placeholder="Sala (ex: Lab 1)" value="Lab 1">
                </div>
                <button type="submit" name="adicionar" class="btn-adicionar">➕ Adicionar Horário</button>
            </form>
            
            <!-- Lista de Horários -->
            <h2>📋 Horários Cadastrados</h2>
            <?php if (empty($horarios)): ?>
                <p style="color:#999; text-align:center; padding:20px;">Nenhum horário cadastrado ainda.</p>
            <?php else: ?>
            <table>
                <thead>
                    <tr>
                        <th>Dia</th>
                        <th>Matéria</th>
                        <th>Professor</th>
                        <th>Início</th>
                        <th>Fim</th>
                        <th>Sala</th>
                        <th>Ações</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($horarios as $h): ?>
                    <tr id="row-<?php echo $h['id']; ?>">
                        <td><?php echo htmlspecialchars($h['dia_semana']); ?></td>
                        <td><?php echo htmlspecialchars($h['materia']); ?></td>
                        <td><?php echo htmlspecialchars($h['professor']); ?></td>
                        <td><?php echo substr($h['horario_inicio'], 0, 5); ?></td>
                        <td><?php echo substr($h['horario_fim'], 0, 5); ?></td>
                        <td><?php echo htmlspecialchars($h['sala']); ?></td>
                        <td class="acoes">
                            <button class="btn-editar" onclick="editar(<?php echo $h['id']; ?>)">✏️ Editar</button>
                            <form method="POST" style="display:inline">
                                <input type="hidden" name="id" value="<?php echo $h['id']; ?>">
                                <button type="submit" name="excluir" class="btn-excluir" onclick="return confirm('Excluir este horário?')">🗑️</button>
                            </form>
                        </td>
                    </tr>
                    <!-- Linha de edição (oculta) -->
                    <tr id="edit-<?php echo $h['id']; ?>" class="edit-row" style="display:none;">
                        <form method="POST">
                            <td><select name="dia" class="select-edit">
                                <?php foreach ($dias as $d): ?>
                                <option value="<?php echo $d; ?>" <?php echo $d == $h['dia_semana'] ? 'selected' : ''; ?>><?php echo $d; ?></option>
                                <?php endforeach; ?>
                            </select></td>
                            <td><input type="text" name="materia" class="input-edit" value="<?php echo htmlspecialchars($h['materia']); ?>"></td>
                            <td><input type="text" name="professor" class="input-edit" value="<?php echo htmlspecialchars($h['professor']); ?>"></td>
                            <td><input type="time" name="horario_inicio" class="input-edit" value="<?php echo $h['horario_inicio']; ?>"></td>
                            <td><input type="time" name="horario_fim" class="input-edit" value="<?php echo $h['horario_fim']; ?>"></td>
                            <td><input type="text" name="sala" class="input-edit" value="<?php echo htmlspecialchars($h['sala']); ?>"></td>
                            <td class="acoes">
                                <input type="hidden" name="id" value="<?php echo $h['id']; ?>">
                                <button type="submit" name="editar" class="btn-salvar">💾 Salvar</button>
                                <button type="button" class="btn-cancelar" onclick="cancelarEditar(<?php echo $h['id']; ?>)">❌ Cancelar</button>
                            </td>
                        </form>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
            <?php endif; ?>
            
            <button onclick="location.href='menu.php'" class="btn-voltar">← Voltar ao Menu</button>
        </div>
    </div>
    
    <script>
    function editar(id) {
        // Esconder a linha normal
        document.getElementById('row-' + id).style.display = 'none';
        // Mostrar a linha de edição
        document.getElementById('edit-' + id).style.display = 'table-row';
    }
    
    function cancelarEditar(id) {
        // Mostrar a linha normal
        document.getElementById('row-' + id).style.display = 'table-row';
        // Esconder a linha de edição
        document.getElementById('edit-' + id).style.display = 'none';
    }
    </script>
</body>
</html>