
from __future__ import annotations
import sys

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/config.py
# ==============================================================================
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    model_names: tuple[str, ...] = ('paper_1d_cnn', 'cnn_gru', 'cnn_lstm', 'cnn_bigru_attention', 'tcn')
    seeds: tuple[int, ...] = (42, 123, 2026)
    outer_folds: int = 5
    epochs: int = 30
    batch_size: int = 32
    patience: int = 6
    learning_rate: float = 0.001
    embedding_dim: int = 128
    projection_dim: int = 64
    brier_weight: float = 0.25
    supcon_weight: float = 0.1
    temperature: float = 0.1
    reload_atol: float = 1e-05

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
DEFAULT_CONFIG = ExperimentConfig()

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/losses.py
# ==============================================================================
import tensorflow as tf
from tensorflow import keras

@keras.utils.register_keras_serializable(package='SHRS')
class BCEBrierLoss(keras.losses.Loss):
    """Binary cross-entropy plus a calibration-sensitive Brier term."""

    def __init__(self, brier_weight: float=0.25, name: str='bce_brier', reduction: str=keras.losses.Reduction.AUTO, **kwargs) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.brier_weight = float(brier_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        brier = tf.square(y_pred - y_true)
        return tf.reduce_mean(bce) + self.brier_weight * tf.reduce_mean(brier)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), 'brier_weight': self.brier_weight}

@keras.utils.register_keras_serializable(package='SHRS')
class ProjectionSupConLoss(keras.losses.Loss):
    """Supervised contrastive loss on the normalized projection branch."""

    def __init__(self, temperature: float=0.1, name: str='supcon_loss', reduction: str=keras.losses.Reduction.AUTO, **kwargs) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.temperature = float(temperature)

    def call(self, y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        z = tf.math.l2_normalize(y_pred, axis=1)
        logits = tf.matmul(z, z, transpose_b=True) / self.temperature
        n = tf.shape(logits)[0]
        eye = tf.eye(n, dtype=logits.dtype)
        not_self = 1.0 - eye
        same = tf.cast(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)), logits.dtype)
        positives = same * not_self
        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        exp_logits = tf.exp(logits) * not_self
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-08)
        positive_count = tf.reduce_sum(positives, axis=1)
        mean_positive = tf.reduce_sum(positives * log_prob, axis=1) / tf.maximum(positive_count, 1.0)
        valid = tf.cast(positive_count > 0, logits.dtype)
        return tf.reduce_sum(-mean_positive * valid) / (tf.reduce_sum(valid) + 1e-08)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), 'temperature': self.temperature}

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/data_io.py
# ==============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
METADATA_COLUMNS = {'sample_id', 'group_id', 'label', 'filename', 'label_name', 'retained_spectra', 'total_spectra'}
FEATURE_PREFIXES = ('x_', 'f_', 'feature_', 'raman_', 'wn_')

def load_ready_dataset(path: str | Path):
    """Return X, y, groups, sample_ids, axis, and the group-ID source.

    For the supplied 20-row table, every row is already one patient spectrum.
    Therefore ``sample_id`` is used as ``group_id`` when the latter is absent.
    Spectral values are not transformed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npz':
        data = np.load(path, allow_pickle=False)
        x = data['X']
        y = data['y']
        groups = data['groups']
        sample_ids = data['sample_ids'] if 'sample_ids' in data.files else None
        axis = data['axis'] if 'axis' in data.files else None
        return (x, y, groups, sample_ids, axis, 'npz:groups')
    if suffix != '.csv':
        raise ValueError('Use a ready numerical .csv or .npz dataset')
    frame = pd.read_csv(path)
    if 'label' not in frame.columns:
        raise ValueError('CSV is missing the required label column')
    if 'group_id' in frame.columns:
        groups = frame['group_id'].astype(str).to_numpy()
        group_source = 'CSV group_id'
    elif 'sample_id' in frame.columns:
        groups = frame['sample_id'].astype(str).to_numpy()
        group_source = 'sample_id fallback (one supplied row per patient)'
    else:
        raise ValueError('CSV must contain group_id or a one-patient-per-row sample_id')
    prefixed = [column for column in frame.columns if any((str(column).lower().startswith(prefix) for prefix in FEATURE_PREFIXES))]
    if prefixed:
        feature_columns = prefixed
    else:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        feature_columns = [column for column in numeric_columns if column not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError('No numeric feature columns were found')
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame['label'].to_numpy()
    if 'sample_id' in frame.columns:
        sample_ids = frame['sample_id'].astype(str).to_numpy()
    else:
        sample_ids = np.array([f'sample_{index:05d}' for index in range(len(frame))], dtype=str)
    axis = np.asarray(feature_columns, dtype=str)
    return (x, y, groups, sample_ids, axis, group_source)

def validate_ready_arrays(x, y, groups, sample_ids=None, required_folds: int=5, group_source: str='provided') -> dict[str, object]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.int32)
    groups = np.asarray(groups).reshape(-1).astype(str)
    if sample_ids is None:
        sample_ids = np.array([f'sample_{index:05d}' for index in range(len(y))], dtype=str)
    else:
        sample_ids = np.asarray(sample_ids).reshape(-1).astype(str)
    errors: list[str] = []
    if x.ndim != 2:
        errors.append(f'X must be 2D (samples, features); got {x.shape}')
    if not len(x) == len(y) == len(groups) == len(sample_ids):
        errors.append('X, y, groups, and sample_ids lengths differ')
    if not np.isfinite(x).all():
        errors.append('X contains NaN or infinity; fix it upstream before AI training')
    if not set(np.unique(y)).issubset({0, 1}):
        errors.append(f'Labels must be binary 0/1; got {np.unique(y)}')
    if pd.Series(sample_ids).duplicated().any():
        errors.append('sample_id values must be unique')
    group_frame = pd.DataFrame({'group': groups, 'label': y})
    mixed = group_frame.groupby('group')['label'].nunique()
    if (mixed > 1).any():
        errors.append('At least one patient/group has mixed labels')
    group_labels = group_frame.groupby('group')['label'].first()
    class_counts = {int(label): int(count) for label, count in group_labels.value_counts().sort_index().items()}
    feasible = len(class_counts) == 2 and min(class_counts.values()) >= required_folds
    if errors:
        raise ValueError('\n'.join(errors))
    return {'n_samples': int(len(y)), 'n_features': int(x.shape[1]), 'n_groups': int(len(group_labels)), 'group_class_counts': class_counts, 'group_id_source': group_source, 'requested_folds': int(required_folds), 'folds_feasible': bool(feasible), 'values_modified_by_loader': False}

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/metrics.py
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, roc_auc_score, roc_curve

def patient_table(y, groups, probabilities) -> pd.DataFrame:
    frame = pd.DataFrame({'group_id': np.asarray(groups).astype(str), 'label': np.asarray(y).astype(int), 'probability': np.asarray(probabilities).reshape(-1)})
    if (frame.groupby('group_id')['label'].nunique() > 1).any():
        raise ValueError('At least one patient/group has mixed labels')
    return frame.groupby('group_id', as_index=False).agg(label=('label', 'first'), probability=('probability', 'mean'), n_spectra=('probability', 'size'))

def choose_threshold(y, probabilities) -> float:
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, probabilities)
    score = true_positive_rate - false_positive_rate
    best = np.flatnonzero(score == np.nanmax(score))
    if len(best) > 1:
        return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])
    return float(thresholds[best[0]])

def calculate_metrics(y, probabilities, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, probabilities)), 'pr_auc': float(average_precision_score(y, probabilities)), 'accuracy': float(accuracy_score(y, predictions)), 'balanced_accuracy': float(balanced_accuracy_score(y, predictions)), 'sensitivity': float(tp / (tp + fn)) if tp + fn else float('nan'), 'specificity': float(tn / (tn + fp)) if tn + fp else float('nan'), 'precision': float(precision_score(y, predictions, zero_division=0)), 'f1': float(f1_score(y, predictions, zero_division=0)), 'brier': float(brier_score_loss(y, probabilities)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/inspection.py
# ==============================================================================
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

def _shape(value) -> str:
    try:
        return str(value.shape)
    except AttributeError:
        try:
            return str([item.shape for item in value])
        except (AttributeError, TypeError):
            return ''

def _purpose(layer) -> str:
    name = layer.name.lower()
    kind = layer.__class__.__name__.lower()
    if 'input' in kind:
        return 'ordered Raman spectrum input'
    if 'conv' in kind and 'residual' in name:
        return 'match residual channel width'
    if 'conv' in kind:
        return 'learn local or dilated spectral patterns'
    if 'batchnormalization' in kind:
        return 'stabilize activation scale during training'
    if 'leakyrelu' in kind:
        return 'nonlinearity that retains a small negative gradient'
    if 'averagepooling' in kind:
        return 'compress spectral length by two'
    if 'bidirectional' in kind:
        return 'read spectral order in both directions'
    if kind in {'gru', 'lstm'}:
        return 'summarize ordered spectral dependencies'
    if 'multiheadattention' in kind:
        return 'relate distant spectral regions with four heads'
    if kind == 'add':
        return 'residual information path'
    if 'layernormalization' in kind:
        return 'normalize attention residual representation'
    if 'globalaveragepooling' in kind:
        return 'convert sequence to one patient-level vector'
    if 'flatten' in kind:
        return 'paper head: 63 x 128 = 8,064 values'
    if name == 'embedding_dense':
        return '128-D representation shared by both supervised heads'
    if name == 'prob':
        return 'sigmoid probability for class 1'
    if name == 'projection_dense':
        return '64-D supervised-contrastive projection'
    if name == 'projection':
        return 'unit-normalize contrastive representation'
    return 'implemented model operation'

def layer_table(model: keras.Model) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, layer in enumerate(model.layers):
        config = layer.get_config()
        rows.append({'index': index, 'name': layer.name, 'type': layer.__class__.__name__, 'input_shape': _shape(layer.input), 'output_shape': _shape(layer.output), 'filters': config.get('filters', ''), 'units': config.get('units', ''), 'kernel_size': config.get('kernel_size', ''), 'strides': config.get('strides', ''), 'dilation_rate': config.get('dilation_rate', ''), 'padding': config.get('padding', ''), 'activation': config.get('activation', ''), 'pool_size': config.get('pool_size', ''), 'dropout': config.get('dropout', config.get('rate', '')), 'return_sequences': config.get('return_sequences', ''), 'num_heads': config.get('num_heads', ''), 'key_dim': config.get('key_dim', ''), 'negative_slope': config.get('alpha', ''), 'parameters': int(layer.count_params()), 'trainable': bool(layer.trainable), 'purpose': _purpose(layer)})
    return pd.DataFrame(rows)

def draw_layer_diagram(table: pd.DataFrame, model_name: str, output_path: str | Path) -> Path:
    """Draw a separate readable diagram; every implemented layer gets a box."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_column = 8
    column_count = math.ceil(len(table) / rows_per_column)
    figure, axis = plt.subplots(figsize=(6.4 * column_count, 2.15 * rows_per_column + 1.4))
    axis.set_xlim(0, column_count)
    axis.set_ylim(0, rows_per_column + 0.9)
    axis.axis('off')
    positions: list[tuple[float, float]] = []
    for sequence, (_, row) in enumerate(table.iterrows()):
        column = sequence // rows_per_column
        within = sequence % rows_per_column
        if column % 2 == 0:
            y = rows_per_column - within - 0.25
        else:
            y = within + 0.75
        x = column + 0.08
        positions.append((x + 0.42, y))
        details = []
        for key in ('filters', 'units', 'kernel_size', 'strides', 'dilation_rate', 'pool_size', 'dropout', 'num_heads', 'key_dim'):
            value = row[key]
            if value != '' and (not pd.isna(value)):
                details.append(f'{key}={value}')
        label = f"{sequence + 1}. {row['name']} ({row['type']})\nOutput: {row['output_shape']} | Params: {row['parameters']}\n{', '.join(details)}\nWhy: {row['purpose']}"
        axis.text(x + 0.42, y, label, ha='center', va='center', fontsize=8, bbox={'boxstyle': 'round,pad=0.55', 'facecolor': '#d9eaf7', 'edgecolor': '#23415b', 'linewidth': 1.1})
    for start, end in zip(positions[:-1], positions[1:]):
        axis.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '->', 'color': '#23415b', 'lw': 1.1})
    axis.set_title(f'{model_name}: every implemented AI layer in execution order', fontsize=16, weight='bold')
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return output_path

def export_architectures(output_dir: str | Path, n_features: int=1011):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, builder in BUILDERS.items():
        model = builder(n_features)
        table = layer_table(model)
        table_path = output_dir / f'{name}_layers.csv'
        table.to_csv(table_path, index=False)
        outputs.extend([table_path, draw_layer_diagram(table, name, output_dir / f'{name}_architecture.png')])
    return outputs

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/common.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x, filters: int, index: int, prefix: str, pool: bool=True):
    name = f'{prefix}{index}'
    x = layers.Conv1D(filters, 3, strides=1, padding='same', use_bias=False, name=f'{name}_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.LeakyReLU(alpha=0.1, name=f'{name}_lrelu')(x)
    if pool:
        x = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid', name=f'{name}_avgpool')(x)
    return x

def supervised_heads(x, config: ExperimentConfig=DEFAULT_CONFIG):
    embedding = layers.Dense(config.embedding_dim, name='embedding_dense')(x)
    embedding = layers.LeakyReLU(alpha=0.1, name='embedding_activation')(embedding)
    probability = layers.Dense(1, activation='sigmoid', name='prob')(embedding)
    projection = layers.Dense(config.projection_dim, name='projection_dense')(embedding)
    projection = layers.UnitNormalization(axis=-1, name='projection')(projection)
    return {'prob': probability, 'projection': projection}

def compile_model(model: keras.Model, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss={'prob': BCEBrierLoss(config.brier_weight), 'projection': ProjectionSupConLoss(config.temperature)}, loss_weights={'prob': 1.0, 'projection': config.supcon_weight}, metrics={'prob': [keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='roc_auc'), keras.metrics.AUC(name='pr_auc', curve='PR')]})
    return model

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/paper_1d_cnn.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def build_paper_1d_cnn(n_features: int=1011, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    inputs = keras.Input((n_features, 1), name='spectrum')
    x = inputs
    for index, filters in enumerate((16, 32, 64, 128), start=1):
        x = conv_block(x, filters, index, 'paper_block', pool=True)
    x = layers.Flatten(name='flatten')(x)
    return keras.Model(inputs, supervised_heads(x, config), name='paper_1d_cnn')

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/cnn_gru.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_gru(n_features: int=1011, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    inputs = keras.Input((n_features, 1), name='spectrum')
    x = conv_block(inputs, 32, 1, 'cnn_gru_block', pool=True)
    x = conv_block(x, 64, 2, 'cnn_gru_block', pool=True)
    x = layers.GRU(64, dropout=0.1, return_sequences=False, name='gru')(x)
    return keras.Model(inputs, supervised_heads(x, config), name='cnn_gru')

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/cnn_lstm.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_lstm(n_features: int=1011, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    inputs = keras.Input((n_features, 1), name='spectrum')
    x = conv_block(inputs, 32, 1, 'cnn_lstm_block', pool=True)
    x = conv_block(x, 64, 2, 'cnn_lstm_block', pool=True)
    x = layers.LSTM(64, dropout=0.1, return_sequences=False, name='lstm')(x)
    return keras.Model(inputs, supervised_heads(x, config), name='cnn_lstm')

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/cnn_bigru_attention.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_bigru_attention(n_features: int=1011, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    inputs = keras.Input((n_features, 1), name='spectrum')
    x = conv_block(inputs, 32, 1, 'cnn_bigru_block', pool=True)
    x = conv_block(x, 64, 2, 'cnn_bigru_block', pool=True)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.1), name='bigru')(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1, name='self_attention')(x, x)
    x = layers.Add(name='attention_residual')([x, attention])
    x = layers.LayerNormalization(name='attention_norm')(x)
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    return keras.Model(inputs, supervised_heads(x, config), name='cnn_bigru_attention')

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/tcn.py
# ==============================================================================
from tensorflow import keras
from tensorflow.keras import layers

def tcn_block(x, filters: int, dilation: int, index: int):
    residual = x
    prefix = f'tcn_block{index}'
    x = layers.Conv1D(filters, 3, padding='same', dilation_rate=dilation, use_bias=False, name=f'{prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{prefix}_bn1')(x)
    x = layers.LeakyReLU(alpha=0.1, name=f'{prefix}_act1')(x)
    x = layers.Conv1D(filters, 3, padding='same', dilation_rate=dilation, use_bias=False, name=f'{prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{prefix}_bn2')(x)
    if int(residual.shape[-1]) != filters:
        residual = layers.Conv1D(filters, 1, padding='same', name=f'{prefix}_residual_projection')(residual)
    x = layers.Add(name=f'{prefix}_add')([x, residual])
    return layers.LeakyReLU(alpha=0.1, name=f'{prefix}_outact')(x)

def build_tcn(n_features: int=1011, config: ExperimentConfig=DEFAULT_CONFIG) -> keras.Model:
    inputs = keras.Input((n_features, 1), name='spectrum')
    x = inputs
    for index, dilation in enumerate((1, 2, 4, 8), start=1):
        x = tcn_block(x, 64, dilation, index)
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    return keras.Model(inputs, supervised_heads(x, config), name='tcn')

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/models/__init__.py
# ==============================================================================
from collections.abc import Callable
from tensorflow import keras
BUILDERS: dict[str, Callable[[int], keras.Model]] = {'paper_1d_cnn': build_paper_1d_cnn, 'cnn_gru': build_cnn_gru, 'cnn_lstm': build_cnn_lstm, 'cnn_bigru_attention': build_cnn_bigru_attention, 'tcn': build_tcn}

def build_model(name: str, n_features: int) -> keras.Model:
    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown model {name!r}; choose from {tuple(BUILDERS)}') from exc
    return builder(int(n_features))
__all__ = ['BUILDERS', 'build_model']

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/reporting.py
# ==============================================================================
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
MODEL_LABELS = {'paper_1d_cnn': 'Paper 1D-CNN', 'cnn_gru': 'CNN-GRU', 'cnn_lstm': 'CNN-LSTM', 'cnn_bigru_attention': 'CNN-BiGRU + Attention', 'tcn': 'TCN'}
QUALITY_METRICS = [('roc_auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC'), ('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'), ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity'), ('precision', 'Precision'), ('f1', 'F1 score')]

def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    if 'rank' in frame:
        return frame.sort_values('rank').reset_index(drop=True)
    return frame.sort_values('model').reset_index(drop=True)

def _labels(models) -> list[str]:
    return [MODEL_LABELS.get(str(name), str(name)) for name in models]

def _finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def readable_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison = _ordered(comparison)
    table = pd.DataFrame({'Rank': comparison['rank'].astype(int), 'Model': _labels(comparison['model']), 'Evaluations': comparison['completed_outer_evaluations'].astype(int), 'Parameters': comparison['parameters'].astype(int)})
    for metric, label in QUALITY_METRICS + [('brier', 'Brier (lower is better)')]:
        table[label] = [f'{mean:.3f} ± {sd:.3f}' for mean, sd in zip(comparison[f'{metric}_mean'], comparison[f'{metric}_sd'])]
    table['Training s'] = [f'{mean:.2f} ± {sd:.2f}' for mean, sd in zip(comparison['training_seconds_mean'], comparison['training_seconds_sd'])]
    table['Inference ms/spectrum'] = [f'{mean:.3f} ± {sd:.3f}' for mean, sd in zip(comparison['inference_ms_per_spectrum_mean'], comparison['inference_ms_per_spectrum_sd'])]
    table['All reload checks'] = np.where(comparison['all_reloads_ok'], 'PASS', 'FAIL')
    return table

def _write_markdown_table(table: pd.DataFrame, path: Path) -> None:

    def clean(value: object) -> str:
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join((clean(column) for column in table.columns)) + ' |', '| ' + ' | '.join(('---' for _ in table.columns)) + ' |']
    lines.extend(('| ' + ' | '.join((clean(value) for value in row)) + ' |' for row in table.itertuples(index=False, name=None)))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def plot_quality_summary(comparison: pd.DataFrame, path: Path) -> None:
    comparison = _ordered(comparison)
    labels = _labels(comparison['model'])
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(labels)))
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    for axis, (metric, title) in zip(axes.flat, QUALITY_METRICS):
        means = comparison[f'{metric}_mean'].to_numpy()
        errors = comparison[f'{metric}_sd'].fillna(0).to_numpy()
        positions = np.arange(len(labels))
        axis.bar(positions, means, color=colors, alpha=0.88)
        axis.errorbar(positions, means, yerr=errors, fmt='none', ecolor='#222222', capsize=4, linewidth=1.2)
        axis.set_title(f'{title} — mean ± SD')
        axis.set_ylim(0, 1.05)
        axis.set_xticks(positions, labels, rotation=28, ha='right')
        axis.grid(axis='y', alpha=0.25)
        axis.set_ylabel('Score (0–1)')
    fig.suptitle('Five-model held-out performance comparison', fontsize=16)
    fig.tight_layout()
    _finish_figure(fig, path)

def plot_calibration_efficiency(comparison: pd.DataFrame, path: Path) -> None:
    comparison = _ordered(comparison)
    labels = _labels(comparison['model'])
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(labels)))
    panels = [('brier', 'Brier score', 'Lower is better'), ('training_seconds', 'Training time', 'Seconds per held-out evaluation'), ('inference_ms_per_spectrum', 'Inference latency', 'Milliseconds per spectrum')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (metric, title, ylabel) in zip(axes, panels):
        means = comparison[f'{metric}_mean'].to_numpy()
        errors = comparison[f'{metric}_sd'].fillna(0).to_numpy()
        positions = np.arange(len(labels))
        axis.bar(positions, means, color=colors, alpha=0.88)
        axis.errorbar(positions, means, yerr=errors, fmt='none', ecolor='#222222', capsize=4, linewidth=1.2)
        axis.set_title(f'{title} — mean ± SD')
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=28, ha='right')
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('Calibration and usability comparison', fontsize=16)
    fig.tight_layout()
    _finish_figure(fig, path)

def plot_fold_distributions(fold_results: pd.DataFrame, path: Path) -> None:
    model_order = fold_results.groupby('model')['roc_auc'].mean().sort_values(ascending=False).index
    labels = _labels(model_order)
    metrics = QUALITY_METRICS + [('brier', 'Brier score')]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [fold_results.loc[fold_results.model == model, metric].dropna().to_numpy() for model in model_order]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.set_title(f'{title} across held-out folds/seeds')
        axis.set_xticklabels(labels, rotation=28, ha='right')
        axis.set_ylabel('Score (0–1)')
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('Evaluation-to-evaluation variability', fontsize=16)
    fig.tight_layout()
    _finish_figure(fig, path)

def plot_confusion_counts(fold_results: pd.DataFrame, path: Path) -> None:
    grouped = fold_results.groupby('model')[['tn', 'fp', 'fn', 'tp']].mean().loc[fold_results.groupby('model')['roc_auc'].mean().sort_values(ascending=False).index]
    labels = _labels(grouped.index)
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(len(labels))
    colors = ['#4c78a8', '#f58518', '#e45756', '#54a24b']
    for column, color in zip(['tn', 'fp', 'fn', 'tp'], colors):
        values = grouped[column].to_numpy()
        axis.bar(positions, values, bottom=bottoms, label=column.upper(), color=color)
        bottoms += values
    axis.set_xticks(positions, labels, rotation=20, ha='right')
    axis.set_ylabel('Mean patient count per held-out evaluation')
    axis.set_title('Mean confusion-matrix composition')
    axis.legend(ncols=4)
    axis.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _finish_figure(fig, path)

def make_comparison_outputs(fold_results: pd.DataFrame, comparison: pd.DataFrame, output_root: str | Path) -> list[Path]:
    output_root = Path(output_root)
    graph_dir = output_root / 'comparison_graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    table = readable_comparison_table(comparison)
    csv_path = output_root / 'model_comparison_readable.csv'
    markdown_path = output_root / 'model_comparison.md'
    table.to_csv(csv_path, index=False)
    _write_markdown_table(table, markdown_path)
    outputs = [csv_path, markdown_path]
    graph_functions = [(plot_quality_summary, graph_dir / 'performance_mean_sd.png', comparison), (plot_calibration_efficiency, graph_dir / 'calibration_efficiency_mean_sd.png', comparison), (plot_fold_distributions, graph_dir / 'held_out_distributions.png', fold_results), (plot_confusion_counts, graph_dir / 'confusion_counts_mean.png', fold_results)]
    for function, path, frame in graph_functions:
        function(frame, path)
        outputs.append(path)
    return outputs

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/training.py
# ==============================================================================
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(int(seed))

def predict_probability(model: keras.Model, x, batch_size: int=64) -> np.ndarray:
    output = model.predict(np.asarray(x, dtype=np.float32)[..., None], batch_size=batch_size, verbose=0)
    if isinstance(output, dict):
        output = output['prob']
    else:
        output = output[model.output_names.index('prob')]
    return np.asarray(output).reshape(-1)

def inner_split(train_indices, y, groups, seed: int):
    train_indices = np.asarray(train_indices)
    partition_y = np.asarray(y)[train_indices]
    partition_groups = np.asarray(groups)[train_indices]
    group_labels = pd.DataFrame({'group': partition_groups, 'label': partition_y}).groupby('group')['label'].first()
    minimum_class = int(group_labels.value_counts().min())
    inner_folds = min(4, minimum_class)
    if inner_folds < 2:
        raise ValueError('Outer training partition needs at least two groups per class for validation')
    splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=int(seed))
    fit_local, validation_local = next(splitter.split(np.zeros(len(train_indices)), partition_y, partition_groups))
    return (train_indices[fit_local], train_indices[validation_local])

def train_one(model_name: str, x, y, groups, sample_ids, train_indices, validation_indices, test_indices, output_dir: str | Path, seed: int, config: ExperimentConfig=DEFAULT_CONFIG, epochs: int | None=None) -> dict[str, object]:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = int(x.shape[1])
    model = compile_model(build_model(model_name, n_features), config)
    architecture = layer_table(model)
    architecture.to_csv(output_dir / 'layers.csv', index=False)
    draw_layer_diagram(architecture, model_name, output_dir / 'architecture.png')
    y_train = np.asarray(y)[train_indices].astype(np.float32).reshape(-1, 1)
    y_validation = np.asarray(y)[validation_indices].astype(np.float32).reshape(-1, 1)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True, min_delta=0.0001)]
    start = time.perf_counter()
    history = model.fit(np.asarray(x)[train_indices][..., None], {'prob': y_train, 'projection': y_train}, validation_data=(np.asarray(x)[validation_indices][..., None], {'prob': y_validation, 'projection': y_validation}), epochs=int(epochs or config.epochs), batch_size=config.batch_size, callbacks=callbacks, verbose=0)
    training_seconds = time.perf_counter() - start
    pd.DataFrame(history.history).to_csv(output_dir / 'history.csv', index=False)
    validation_probability = predict_probability(model, x[validation_indices], config.batch_size)
    validation_patient = patient_table(y[validation_indices], groups[validation_indices], validation_probability)
    threshold = choose_threshold(validation_patient.label, validation_patient.probability)
    inference_start = time.perf_counter()
    test_probability = predict_probability(model, x[test_indices], config.batch_size)
    inference_seconds = time.perf_counter() - inference_start
    test_patient = patient_table(y[test_indices], groups[test_indices], test_probability)
    metrics = calculate_metrics(test_patient.label, test_patient.probability, threshold)
    spectrum_predictions = pd.DataFrame({'sample_id': np.asarray(sample_ids)[test_indices], 'group_id': np.asarray(groups)[test_indices], 'label': np.asarray(y)[test_indices], 'probability': test_probability})
    spectrum_predictions.to_csv(output_dir / 'test_spectrum_predictions.csv', index=False)
    test_patient.to_csv(output_dir / 'test_patient_predictions.csv', index=False)
    fixed = np.asarray(x)[test_indices[:min(8, len(test_indices))]][..., None]
    probability_before = predict_probability(model, fixed[..., 0])
    keras_path = output_dir / 'model.keras'
    weights_path = output_dir / 'model.weights.h5'
    portable_path = output_dir / 'model.portable.h5'
    model.save(keras_path)
    model.save_weights(weights_path)
    model.save_weights(portable_path, save_format='h5')
    loaded = keras.models.load_model(keras_path, compile=False)
    probability_keras = predict_probability(loaded, fixed[..., 0])
    rebuilt = build_model(model_name, n_features)
    rebuilt.load_weights(weights_path)
    probability_weights = predict_probability(rebuilt, fixed[..., 0])
    keras_difference = float(np.max(np.abs(probability_before - probability_keras)))
    weights_difference = float(np.max(np.abs(probability_before - probability_weights)))
    reload_ok = bool(keras_difference <= config.reload_atol and weights_difference <= config.reload_atol)
    report: dict[str, object] = {'model': model_name, 'threshold': float(threshold), 'training_seconds': float(training_seconds), 'inference_ms_per_spectrum': float(1000.0 * inference_seconds / max(len(test_indices), 1)), 'epochs_ran': int(len(history.history['loss'])), 'parameters': int(model.count_params()), 'reload_keras_max_abs': keras_difference, 'reload_weights_max_abs': weights_difference, 'reload_ok': reload_ok}
    report.update(metrics)
    (output_dir / 'run_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    keras.backend.clear_session()
    return report

def run_experiment(x, y, groups, sample_ids=None, output_root: str | Path='artifacts/full_run', config: ExperimentConfig=DEFAULT_CONFIG, seeds=None, outer_folds: int | None=None, epochs: int | None=None, model_names=None, group_source: str='provided'):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    groups = np.asarray(groups).reshape(-1).astype(str)
    if sample_ids is None:
        sample_ids = np.array([f'sample_{index:05d}' for index in range(len(y))], dtype=str)
    else:
        sample_ids = np.asarray(sample_ids).reshape(-1).astype(str)
    folds = int(outer_folds or config.outer_folds)
    audit = validate_ready_arrays(x, y, groups, sample_ids, required_folds=folds, group_source=group_source)
    if not audit['folds_feasible']:
        raise ValueError(f"Requested {folds}-fold group CV is impossible; counts={audit['group_class_counts']}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    results: list[dict[str, object]] = []
    run_seeds = list(seeds or config.seeds)
    names = list(model_names or config.model_names)
    for repeat, seed in enumerate(run_seeds, start=1):
        outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        for fold, (outer_train, test_indices) in enumerate(outer.split(x, y, groups), start=1):
            train_indices, validation_indices = inner_split(outer_train, y, groups, seed + fold)
            split = np.full(len(y), 'train', dtype=object)
            split[validation_indices] = 'validation'
            split[test_indices] = 'test'
            pd.DataFrame({'sample_index': np.arange(len(y)), 'sample_id': sample_ids, 'group_id': groups, 'label': y, 'split': split}).to_csv(output_root / f'split_repeat{repeat}_fold{fold}.csv', index=False)
            for model_index, name in enumerate(names):
                model_output = output_root / f'repeat_{repeat}_seed_{seed}' / f'fold_{fold}' / name
                report = train_one(name, x, y, groups, sample_ids, train_indices, validation_indices, test_indices, model_output, seed + fold * 100 + model_index, config, epochs)
                report.update({'repeat': repeat, 'repeat_seed': seed, 'fold': fold, 'n_train': int(len(train_indices)), 'n_validation': int(len(validation_indices)), 'n_test': int(len(test_indices))})
                results.append(report)
                pd.DataFrame(results).to_csv(output_root / 'fold_results.csv', index=False)
                print(f"{name}: repeat={repeat}/{len(run_seeds)} fold={fold}/{folds} auc={report['roc_auc']:.3f}", flush=True)
    fold_results = pd.DataFrame(results)
    metric_columns = ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'brier', 'training_seconds', 'inference_ms_per_spectrum', 'epochs_ran']
    summary_rows: list[dict[str, object]] = []
    for name, frame in fold_results.groupby('model'):
        row: dict[str, object] = {'model': name, 'completed_outer_evaluations': int(len(frame)), 'parameters': int(frame.parameters.iloc[0]), 'all_reloads_ok': bool(frame.reload_ok.all())}
        for metric in metric_columns:
            row[f'{metric}_mean'] = float(frame[metric].mean())
            row[f'{metric}_sd'] = float(frame[metric].std(ddof=1))
        summary_rows.append(row)
    comparison = pd.DataFrame(summary_rows).sort_values(['roc_auc_mean', 'roc_auc_sd'], ascending=[False, True])
    comparison.insert(0, 'rank', np.arange(1, len(comparison) + 1))
    comparison.to_csv(output_root / 'model_comparison.csv', index=False)
    make_comparison_outputs(fold_results, comparison, output_root)
    return (output_root, fold_results, comparison)

def make_smoke_data(n_groups_per_class: int=6, spectra_per_group: int=2, n_features: int=64, seed: int=0):
    generator = np.random.default_rng(seed)
    x_values: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    axis = np.linspace(0, 1, n_features)
    sample_index = 0
    for label in (0, 1):
        for group_index in range(n_groups_per_class):
            group = f'class{label}_patient{group_index:02d}'
            base = np.sin(2 * np.pi * (2 + label) * axis)
            base += label * 0.4 * np.exp(-((axis - 0.65) / 0.08) ** 2)
            base += generator.normal(0, 0.08, n_features)
            for _ in range(spectra_per_group):
                x_values.append(base + generator.normal(0, 0.05, n_features))
                labels.append(label)
                groups.append(group)
                sample_ids.append(f'smoke_{sample_index:04d}')
                sample_index += 1
    return (np.asarray(x_values, dtype=np.float32), np.asarray(labels), np.asarray(groups), np.asarray(sample_ids))

# ==============================================================================
# SOURCE SECTION: src/shrs_ai/cli.py
# ==============================================================================
import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

def _require_python_311() -> None:
    if platform.python_version_tuple()[:2] != ('3', '11'):
        raise RuntimeError(f'Python 3.11 is required; current interpreter is {platform.python_version()}')

def _dataset(path: Path):
    x, y, groups, sample_ids, axis, group_source = load_ready_dataset(path)
    audit = validate_ready_arrays(x, y, groups, sample_ids, required_folds=DEFAULT_CONFIG.outer_folds, group_source=group_source)
    audit['dataset_path'] = str(path.resolve())
    audit['dataset_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return (x, y, groups, sample_ids, axis, group_source, audit)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SHRS Raman AI-only experiment')
    subparsers = parser.add_subparsers(dest='command', required=True)
    inspect_parser = subparsers.add_parser('inspect', help='export exact tables and separate diagrams for all five models')
    inspect_parser.add_argument('--output', type=Path, default=Path('artifacts/architectures'))
    inspect_parser.add_argument('--features', type=int, default=1011)
    audit_parser = subparsers.add_parser('audit', help='validate ready numerical input')
    audit_parser.add_argument('--data', type=Path, required=True)
    smoke_parser = subparsers.add_parser('smoke', help='train/save/reload all five models on synthetic code-test data')
    smoke_parser.add_argument('--output', type=Path, default=Path('artifacts/smoke'))
    run_parser = subparsers.add_parser('run', help='run five models x five folds x three seeds')
    run_parser.add_argument('--data', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, default=None)
    run_parser.add_argument('--confirm-labels', action='store_true')
    run_parser.add_argument('--confirm-groups', action='store_true')
    return parser

def main(argv: list[str] | None=None) -> None:
    _require_python_311()
    args = build_parser().parse_args(argv)
    if args.command == 'inspect':
        outputs = export_architectures(args.output, args.features)
        print(json.dumps([str(path) for path in outputs], indent=2))
        return
    if args.command == 'audit':
        *_, audit = _dataset(args.data)
        print(json.dumps(audit, indent=2))
        return
    if args.command == 'smoke':
        x, y, groups, sample_ids = make_smoke_data(seed=2026)
        root = args.output / datetime.now().strftime('run_%Y%m%d_%H%M%S')
        root, fold_results, comparison = run_experiment(x, y, groups, sample_ids, output_root=root, seeds=[42], outer_folds=2, epochs=1, group_source='synthetic smoke-test groups')
        if len(fold_results) != 10 or not fold_results.reload_ok.all():
            raise AssertionError('Smoke test did not complete 10 reload-equivalent fits')
        print('PASS: 10 synthetic smoke fits completed and reloaded')
        print(comparison.to_string(index=False))
        return
    if not args.confirm_labels or not args.confirm_groups:
        raise RuntimeError('Full supervised training requires --confirm-labels and --confirm-groups')
    x, y, groups, sample_ids, _, group_source, audit = _dataset(args.data)
    if not audit['folds_feasible']:
        raise RuntimeError(f'Five-fold group CV is not feasible: {audit}')
    output = args.output or Path('artifacts', datetime.now().strftime('full_run_%Y%m%d_%H%M%S'))
    root, fold_results, comparison = run_experiment(x, y, groups, sample_ids, output_root=output, seeds=DEFAULT_CONFIG.seeds, outer_folds=DEFAULT_CONFIG.outer_folds, epochs=DEFAULT_CONFIG.epochs, group_source=group_source)
    counts = fold_results.groupby('model').size()
    if len(fold_results) != 75 or not (counts == 15).all():
        raise AssertionError(f'Expected 75 evaluations and 15/model; got {counts}')
    if not fold_results.reload_ok.all():
        raise AssertionError('At least one saved-model reload check failed')
    package = shutil.make_archive(str(root), 'zip', root_dir=root)
    print('FULL RUN COMPLETE')
    print(json.dumps(audit, indent=2))
    print(comparison.to_string(index=False))
    print(f'RESULTS_ZIP={package}')

def all_in_one_entry() -> None:
    """One-click VS Code default: execute the complete 75-evaluation run."""
    if len(sys.argv) == 1:
        project = Path(__file__).resolve().parent
        sys.argv.extend([
            "run",
            "--data",
            str(project / "data" / "processed_samples.csv"),
            "--output",
            str(project / "artifacts" / "all_in_one_full_run"),
            "--confirm-labels",
            "--confirm-groups",
        ])
    main()

if __name__ == "__main__":
    all_in_one_entry()
