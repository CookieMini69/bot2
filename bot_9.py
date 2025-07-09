import time
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
API_KEY = 'Ow2YOkoa1eQbuzojAUM7QSpOABzRbSpH4WtXuNSHz7fNbxFVaeGeZooNBgq11z8h'
API_SECRET = 'O7AIg3GNSy5eYpAuYiKg4TQotlQe2y2wlKHeqcseq6n6DY4XVceGjnJzQB7GdXgz'

# Configuration du trading
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
INTERVAL = '5m'
LOOKBACK_PERIOD = 300
RISK_PERCENT = 0.02
MAX_POSITIONS = 2
SLEEP_TIME = 1

# Configuration des timeframes multiples
TIMEFRAMES = {
    '5m': '5m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h'
}

# Paramètres de stratégie
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 200
SMA_TREND = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Gestion des risques avancée
INITIAL_STOP_LOSS_PERCENT = 0.03
TRAILING_STOP_PERCENT = 0.02
FIRST_TAKE_PROFIT_PERCENT = 0.06
FINAL_TAKE_PROFIT_PERCENT = 0.12
PYRAMIDING_LEVELS = 3

# Filtres de volatilité dynamiques
MIN_ATR_PERCENT = 0.002
MAX_ATR_PERCENT = 0.08
VOLATILITY_MULTIPLIER = 2.0

# Configuration ML
ML_LOOKBACK = 1000
ML_RETRAIN_INTERVAL = 24
ML_CONFIDENCE_THRESHOLD = 0.7

# Notifications
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'your_email@gmail.com',
    'password': 'your_app_password',
    'to_email': 'destination@gmail.com'
}

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTradingBot:
    def __init__(self, testnet=True):
        self.client = Client(API_KEY, API_SECRET, testnet=testnet)
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.last_signals: Dict[str, str] = {}
        self.cooldown: Dict[str, float] = {}
        self.ml_models: Dict[str, LogisticRegression] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.last_ml_training: Dict[str, datetime] = {}
        self.symbol_info_cache: Dict[str, Dict] = {}
        self.backtest_mode = False
        self.backtest_data = {}
        
        # Vérification de la connexion
        try:
            self.client.get_account()
            logger.info("Connexion Binance réussie")
            self._cache_symbol_info()
        except Exception as e:
            logger.error(f"Erreur de connexion Binance: {e}")
            raise
    
    def _cache_symbol_info(self):
        """Met en cache les informations des symboles avec gestion d'erreurs améliorée"""
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] in SYMBOLS:
                    self.symbol_info_cache[symbol_info['symbol']] = symbol_info
                    logger.info(f"Cache info pour {symbol_info['symbol']}: {symbol_info.get('filters', [])}")
            logger.info("Informations des symboles mises en cache")
        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache des symboles: {e}")
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Récupère les informations d'un symbole avec fallback"""
        try:
            if symbol in self.symbol_info_cache:
                return self.symbol_info_cache[symbol]
            
            # Fallback : récupération directe
            info = self.client.get_symbol_info(symbol)
            self.symbol_info_cache[symbol] = info
            return info
            
        except Exception as e:
            logger.error(f"Erreur récupération info symbole {symbol}: {e}")
            return {}
    
    def get_min_qty(self, symbol: str) -> float:
        """Récupère la quantité minimale pour un symbole avec gestion d'erreurs robuste"""
        try:
            info = self.get_symbol_info(symbol)
            if not info or 'filters' not in info:
                logger.warning(f"Pas d'info filtres pour {symbol}, utilisation valeur par défaut")
                return 0.001
            
            # Recherche du filtre LOT_SIZE
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = float(filter_info['minQty'])
                    logger.info(f"MinQty pour {symbol}: {min_qty}")
                    return min_qty
            
            # Recherche alternative du filtre MARKET_LOT_SIZE
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'MARKET_LOT_SIZE':
                    min_qty = float(filter_info['minQty'])
                    logger.info(f"Market MinQty pour {symbol}: {min_qty}")
                    return min_qty
            
            logger.warning(f"Filtre LOT_SIZE non trouvé pour {symbol}")
            return 0.001
            
        except Exception as e:
            logger.error(f"Erreur get_min_qty pour {symbol}: {e}")
            return 0.001
    
    def get_step_size(self, symbol: str) -> float:
        """Récupère le step size pour un symbole avec gestion d'erreurs robuste"""
        try:
            info = self.get_symbol_info(symbol)
            if not info or 'filters' not in info:
                logger.warning(f"Pas d'info filtres pour {symbol}, utilisation valeur par défaut")
                return 0.001
            
            # Recherche du filtre LOT_SIZE
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    logger.info(f"StepSize pour {symbol}: {step_size}")
                    return step_size
            
            # Recherche alternative du filtre MARKET_LOT_SIZE
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'MARKET_LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    logger.info(f"Market StepSize pour {symbol}: {step_size}")
                    return step_size
            
            logger.warning(f"Filtre LOT_SIZE non trouvé pour {symbol}")
            return 0.001
            
        except Exception as e:
            logger.error(f"Erreur get_step_size pour {symbol}: {e}")
            return 0.001
    
    def get_price_filter(self, symbol: str) -> Dict[str, float]:
        """Récupère les filtres de prix pour un symbole"""
        try:
            info = self.get_symbol_info(symbol)
            if not info or 'filters' not in info:
                return {'minPrice': 0.01, 'maxPrice': 1000000, 'tickSize': 0.01}
            
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'PRICE_FILTER':
                    return {
                        'minPrice': float(filter_info['minPrice']),
                        'maxPrice': float(filter_info['maxPrice']),
                        'tickSize': float(filter_info['tickSize'])
                    }
            
            return {'minPrice': 0.01, 'maxPrice': 1000000, 'tickSize': 0.01}
            
        except Exception as e:
            logger.error(f"Erreur get_price_filter pour {symbol}: {e}")
            return {'minPrice': 0.01, 'maxPrice': 1000000, 'tickSize': 0.01}
    
    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        """Ajuste la quantité selon les règles du symbole"""
        try:
            min_qty = self.get_min_qty(symbol)
            step_size = self.get_step_size(symbol)
            
            # Vérification quantité minimale
            if quantity < min_qty:
                logger.warning(f"Quantité {quantity} < min {min_qty} pour {symbol}")
                return 0
            
            # Ajustement selon step_size
            if step_size > 0:
                # Calcul du nombre de steps
                steps = int(quantity / step_size)
                adjusted_quantity = steps * step_size
                
                # Vérification que la quantité ajustée reste valide
                if adjusted_quantity >= min_qty:
                    return adjusted_quantity
                else:
                    # Essai avec un step de plus
                    return (steps + 1) * step_size
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur adjust_quantity pour {symbol}: {e}")
            return 0
    
    def send_notification(self, message: str, trade_type: str = 'INFO'):
        """Envoie une notification par Telegram et email"""
        try:
            # Notification Telegram
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN != 'YOUR_TELEGRAM_BOT_TOKEN':
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                emoji = "🚀" if trade_type == 'BUY' else "🔴" if trade_type == 'SELL' else "🤖"
                data = {
                    'chat_id': TELEGRAM_CHAT_ID,
                    'text': f"{emoji} Trading Bot - {trade_type}\n{message}",
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    logger.info("Notification Telegram envoyée")
                else:
                    logger.warning(f"Erreur notification Telegram: {response.status_code}")
            
            # Notification email
            if (EMAIL_CONFIG['email'] and EMAIL_CONFIG['password'] and 
                EMAIL_CONFIG['email'] != 'your_email@gmail.com'):
                
                msg = MIMEMultipart()
                msg['From'] = EMAIL_CONFIG['email']
                msg['To'] = EMAIL_CONFIG['to_email']
                msg['Subject'] = f"Trading Bot - {trade_type}"
                
                body = f"""
                Trading Bot Notification
                
                Type: {trade_type}
                Message: {message}
                Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
                server.starttls()
                server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
                server.send_message(msg)
                server.quit()
                logger.info("Notification email envoyée")
                
        except Exception as e:
            logger.error(f"Erreur notification: {e}")
    
    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        """Récupère les données de marché avec gestion d'erreurs"""
        try:
            if self.backtest_mode and symbol in self.backtest_data:
                return self.backtest_data[symbol].copy()
            
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                logger.warning(f"Pas de données reçues pour {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Conversion des types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Nettoyage des données
            df = df.dropna()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Données récupérées pour {symbol}: {len(df)} lignes")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Erreur API Binance pour {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur inattendue pour {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calcule l'EMA"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calcule la SMA"""
        return data.rolling(window=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Éviter la division par zéro
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Valeur neutre si problème
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule les bandes de Bollinger"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule le MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcule l'ATR"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0)
    
    def calculate_multi_timeframe_trend(self, symbol: str) -> Dict[str, float]:
        """Calcule la tendance sur plusieurs timeframes"""
        trends = {}
        
        for tf_name, tf_interval in TIMEFRAMES.items():
            try:
                df = self.get_klines(symbol, tf_interval, 100)
                if df.empty or len(df) < 50:
                    trends[tf_name] = 0
                    continue
                
                df['ema_fast'] = self.calculate_ema(df['close'], 12)
                df['ema_slow'] = self.calculate_ema(df['close'], 26)
                df['ema_trend'] = self.calculate_ema(df['close'], 50)
                
                current = df.iloc[-1]
                trend_score = 0
                
                # Tendance générale
                if current['close'] > current['ema_trend']:
                    trend_score += 0.5
                else:
                    trend_score -= 0.5
                
                # Momentum court terme
                if current['ema_fast'] > current['ema_slow']:
                    trend_score += 0.5
                else:
                    trend_score -= 0.5
                
                trends[tf_name] = trend_score
                
            except Exception as e:
                logger.error(f"Erreur calcul tendance {tf_name} pour {symbol}: {e}")
                trends[tf_name] = 0
        
        return trends
    
    def calculate_dynamic_volatility_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule des seuils dynamiques basés sur la volatilité"""
        if len(df) < 20:
            return {
                'stop_loss': INITIAL_STOP_LOSS_PERCENT,
                'take_profit': FIRST_TAKE_PROFIT_PERCENT,
                'position_size_multiplier': 1.0
            }
        
        try:
            current_atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price == 0 or pd.isna(current_atr):
                atr_percent = 0.02
            else:
                atr_percent = current_atr / current_price
            
            # Normalisation de la volatilité
            atr_percent = max(0.005, min(0.1, atr_percent))  # Bornes
            
            # Facteur de volatilité
            volatility_factor = max(0.5, min(2.0, atr_percent / 0.02))
            
            return {
                'stop_loss': INITIAL_STOP_LOSS_PERCENT * volatility_factor,
                'take_profit': FIRST_TAKE_PROFIT_PERCENT * volatility_factor,
                'position_size_multiplier': 1.0 / volatility_factor
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul seuils volatilité: {e}")
            return {
                'stop_loss': INITIAL_STOP_LOSS_PERCENT,
                'take_profit': FIRST_TAKE_PROFIT_PERCENT,
                'position_size_multiplier': 1.0
            }
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features pour le modèle ML"""
        if len(df) < 50:
            return pd.DataFrame()
        
        try:
            features = pd.DataFrame(index=df.index)
            
            # Features techniques
            features['rsi'] = df['rsi']
            features['macd_signal'] = df['macd'] - df['macd_signal']
            
            # Position dans les bandes de Bollinger
            bb_width = df['bb_upper'] - df['bb_lower']
            features['bb_position'] = np.where(bb_width != 0, 
                                             (df['close'] - df['bb_lower']) / bb_width, 
                                             0.5)
            
            # Signal EMA
            features['ema_signal'] = (df['ema_fast'] - df['ema_slow']) / df['close']
            
            # Ratio de volume
            features['volume_ratio'] = np.where(df['volume_ma'] != 0,
                                              df['volume'] / df['volume_ma'],
                                              1.0)
            
            # ATR pourcentage
            features['atr_percent'] = df['atr_percent']
            
            # Features de momentum
            features['price_change_1'] = df['close'].pct_change(1)
            features['price_change_5'] = df['close'].pct_change(5)
            features['price_change_10'] = df['close'].pct_change(10)
            
            # Features de volatilité
            features['volatility_5'] = df['close'].rolling(5).std() / df['close']
            features['volatility_20'] = df['close'].rolling(20).std() / df['close']
            
            # Features de tendance
            features['trend_strength'] = (df['close'] - df['ema_trend']) / df['ema_trend']
            
            # Nettoyage des données
            features = features.replace([np.inf, -np.inf], 0)
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur préparation features ML: {e}")
            return pd.DataFrame()
    
    def create_ml_labels(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """Crée les labels pour le modèle ML"""
        if len(df) < forward_periods + 1:
            return pd.Series()
        
        try:
            # Label: 1 si le prix monte de plus de 1% dans les prochaines périodes
            future_returns = df['close'].shift(-forward_periods) / df['close'] - 1
            labels = (future_returns > 0.0025).astype(int)
            
            return labels[:-forward_periods]
            
        except Exception as e:
            logger.error(f"Erreur création labels ML: {e}")
            return pd.Series()
    
    def train_ml_model(self, symbol: str):
        """Entraîne le modèle ML pour un symbole"""
        try:
            logger.info(f"Entraînement du modèle ML pour {symbol}")
            
            # Récupération des données historiques
            df = self.get_klines(symbol, INTERVAL, ML_LOOKBACK)
            if df.empty or len(df) < ML_LOOKBACK // 2:
                logger.warning(f"Pas assez de données pour entraîner le modèle ML pour {symbol}")
                return
            
            # Calcul des indicateurs
            df = self.calculate_indicators(df)
            
            # Préparation des features et labels
            features = self.prepare_ml_features(df)
            labels = self.create_ml_labels(df)
            
            if features.empty or labels.empty:
                logger.warning(f"Erreur dans la préparation des données ML pour {symbol}")
                return
            
            # Alignement des données
            min_length = min(len(features), len(labels))
            features = features.iloc[:min_length]
            labels = labels.iloc[:min_length]
            
            # Vérification des données
            if len(features) < 100:
                logger.warning(f"Pas assez de données alignées pour {symbol}: {len(features)}")
                return
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Normalisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entraînement du modèle
            model = LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)
            
            # Évaluation
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Prédictions pour rapport détaillé
            y_pred = model.predict(X_test_scaled)
            
            logger.info(f"Modèle ML entraîné pour {symbol}:")
            logger.info(f"  - Score train: {train_score:.3f}")
            logger.info(f"  - Score test: {test_score:.3f}")
            logger.info(f"  - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            
            # Sauvegarde du modèle
            self.ml_models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_ml_training[symbol] = datetime.now()
            
            logger.info(f"Modèle ML sauvegardé pour {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur entraînement ML pour {symbol}: {e}")
    
    def get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Obtient une prédiction du modèle ML"""
        try:
            if symbol not in self.ml_models or symbol not in self.scalers:
                return 0.5, 0.5
            
            features = self.prepare_ml_features(df)
            if features.empty:
                return 0.5, 0.5
            
            # Utilisation des dernières données
            last_features = features.iloc[-1].values.reshape(1, -1)

            
            # Normalisation
            last_features_scaled = self.scalers[symbol].transform(last_features)
            
            # Prédiction
            prediction = self.ml_models[symbol].predict(last_features_scaled)[0]
            probabilities = self.ml_models[symbol].predict_proba(last_features_scaled)[0]
            probability = probabilities[1]  # Probabilité de la classe positive
            
            return float(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Erreur prédiction ML pour {symbol}: {e}")
            return 0.5, 0.5
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques"""
        if len(df) < max(EMA_SLOW, RSI_PERIOD, BOLLINGER_PERIOD, MACD_SLOW, EMA_TREND):
            logger.warning(f"Pas assez de données pour calculer les indicateurs: {len(df)}")
            return df
        
        try:
            # EMAs
            df['ema_fast'] = self.calculate_ema(df['close'], EMA_FAST)
            df['ema_slow'] = self.calculate_ema(df['close'], EMA_SLOW)
            df['ema_trend'] = self.calculate_ema(df['close'], EMA_TREND)
            df['sma_trend'] = self.calculate_sma(df['close'], SMA_TREND)
            
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'], RSI_PERIOD)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
                df['close'], BOLLINGER_PERIOD, BOLLINGER_STD
            )
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(
                df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL
            )
            
            # Volume Moving Average
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # ATR pour volatilité
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], 14)
            df['atr_percent'] = np.where(df['close'] != 0, df['atr'] / df['close'], 0)
            
            # Nettoyage final
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
            return df
    
    def get_advanced_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[str, float]:
        """Génère un signal avancé avec score de confiance"""
        if len(df) < 2:
            return 'HOLD', 0.5
        
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Vérification des données
            # Vérification des données essentielles
            required_fields = ['ema_fast', 'ema_slow', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_percent']
            for field in required_fields:
                if field not in df.columns or pd.isna(current[field]):
                    logger.warning(f"Données manquantes pour {field}")
                    return 'HOLD', 0.5
            
            # Calcul des tendances multi-timeframes
            trends = self.calculate_multi_timeframe_trend(symbol)
            
            # Score de tendance globale
            trend_score = sum(trends.values()) / len(trends) if trends else 0
            
            # Prédiction ML
            ml_prediction, ml_probability = self.get_ml_prediction(symbol, df)
            
            # Calcul des seuils dynamiques
            dynamic_thresholds = self.calculate_dynamic_volatility_thresholds(df)
            
            # === SIGNAUX TECHNIQUES ===
            signals = []
            
            # 1. Signal EMA (pondération : 0.25)
            if current['ema_fast'] > current['ema_slow'] and previous['ema_fast'] <= previous['ema_slow']:
                signals.append(('BUY', 0.25))
            elif current['ema_fast'] < current['ema_slow'] and previous['ema_fast'] >= previous['ema_slow']:
                signals.append(('SELL', 0.25))
            
            # 2. Signal RSI (pondération : 0.20)
            if current['rsi'] < RSI_OVERSOLD and previous['rsi'] >= RSI_OVERSOLD:
                signals.append(('BUY', 0.20))
            elif current['rsi'] > RSI_OVERBOUGHT and previous['rsi'] <= RSI_OVERBOUGHT:
                signals.append(('SELL', 0.20))
            
            # 3. Signal MACD (pondération : 0.20)
            if current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
                signals.append(('BUY', 0.20))
            elif current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
                signals.append(('SELL', 0.20))
            
            # 4. Signal Bollinger Bands (pondération : 0.15)
            if current['close'] < current['bb_lower'] and previous['close'] >= previous['bb_lower']:
                signals.append(('BUY', 0.15))
            elif current['close'] > current['bb_upper'] and previous['close'] <= previous['bb_upper']:
                signals.append(('SELL', 0.15))
            
            # 5. Signal de tendance multi-timeframes (pondération : 0.10)
            if trend_score > 0.5:
                signals.append(('BUY', 0.10))
            elif trend_score < -0.5:
                signals.append(('SELL', 0.10))
            
            # 6. Signal ML (pondération : 0.10)
            if ml_prediction == 1 and ml_probability > ML_CONFIDENCE_THRESHOLD:
                signals.append(('BUY', 0.10))
            elif ml_prediction == 0 and ml_probability < (1 - ML_CONFIDENCE_THRESHOLD):
                signals.append(('SELL', 0.10))
            
            # === CALCUL DU SCORE FINAL ===
            buy_score = sum(weight for signal, weight in signals if signal == 'BUY')
            sell_score = sum(weight for signal, weight in signals if signal == 'SELL')
            
            # Filtrage par volatilité
            if current['atr_percent'] < MIN_ATR_PERCENT:
                logger.info(f"Volatilité trop faible pour {symbol}: {current['atr_percent']:.4f}")
                return 'HOLD', 0.5
            
            if current['atr_percent'] > MAX_ATR_PERCENT:
                logger.info(f"Volatilité trop élevée pour {symbol}: {current['atr_percent']:.4f}")
                return 'HOLD', 0.5
            
            # Filtrage par tendance principale
            if current['close'] > current['ema_trend']:
                main_trend = 'UP'
            elif current['close'] < current['ema_trend']:
                main_trend = 'DOWN'
            else:
                main_trend = 'NEUTRAL'
            
            # Décision finale
            confidence = abs(buy_score - sell_score)
            
            if buy_score > sell_score and buy_score > 0.4 and main_trend in ['UP', 'NEUTRAL']:
                return 'BUY', min(confidence, 1.0)
            elif sell_score > buy_score and sell_score > 0.4 and main_trend in ['DOWN', 'NEUTRAL']:
                return 'SELL', min(confidence, 1.0)
            else:
                return 'HOLD', confidence
                
        except Exception as e:
            logger.error(f"Erreur calcul signal avancé pour {symbol}: {e}")
            return 'HOLD', 0.5
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calcule la taille de position avec money management avancé"""
        try:
            # Récupération du solde
            account = self.client.get_account()
            free_balance = float(account['balances'][0]['free'])  # Assume USDT
            
            # Recherche du solde USDT
            usdt_balance = 0
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            if usdt_balance == 0:
                logger.warning("Pas de solde USDT disponible")
                return 0
            
            # Calcul du montant de base
            base_amount = usdt_balance * RISK_PERCENT
            
            # Récupération des données pour calculer la volatilité
            df = self.get_klines(symbol, INTERVAL, 50)
            if df.empty:
                logger.warning(f"Pas de données pour calculer la position pour {symbol}")
                return 0
            
            # Calcul des seuils dynamiques
            dynamic_thresholds = self.calculate_dynamic_volatility_thresholds(df)
            
            # Ajustement selon la force du signal
            signal_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5 à 1.0
            
            # Ajustement selon la volatilité
            volatility_multiplier = dynamic_thresholds.get('position_size_multiplier', 1.0)
            
            # Calcul du montant final
            final_amount = base_amount * signal_multiplier * volatility_multiplier
            
            # Limitation du montant maximum
            max_amount = usdt_balance * 0.1  # Max 10% du solde
            final_amount = min(final_amount, max_amount)
            
            # Récupération du prix actuel
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Calcul de la quantité
            quantity = final_amount / current_price
            
            # Ajustement selon les règles du symbole
            adjusted_quantity = self.adjust_quantity(symbol, quantity)
            
            logger.info(f"Position calculée pour {symbol}:")
            logger.info(f"  - Montant: {final_amount:.2f} USDT")
            logger.info(f"  - Prix: {current_price:.6f}")
            logger.info(f"  - Quantité brute: {quantity:.6f}")
            logger.info(f"  - Quantité ajustée: {adjusted_quantity:.6f}")
            
            return adjusted_quantity
            
        except Exception as e:
            logger.error(f"Erreur calcul taille position pour {symbol}: {e}")
            return 0
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET') -> Optional[Dict]:
        """Place un ordre avec gestion d'erreurs robuste"""
        try:
            if quantity <= 0:
                logger.warning(f"Quantité invalide pour {symbol}: {quantity}")
                return None
            
            # Vérification finale de la quantité
            adjusted_quantity = self.adjust_quantity(symbol, quantity)
            if adjusted_quantity <= 0:
                logger.warning(f"Quantité ajustée invalide pour {symbol}: {adjusted_quantity}")
                return None
            
            # Vérification des fonds
            account = self.client.get_account()
            
            if self.backtest_mode:
                # Simulation d'ordre en backtest
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                order = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'executedQty': str(adjusted_quantity),
                    'fills': [{'price': ticker['price'], 'qty': str(adjusted_quantity)}],
                    'status': 'FILLED',
                    'orderId': int(time.time() * 1000)  # ID simulé
                }
                return order
            
            # Placement de l'ordre réel
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=adjusted_quantity
            )
            
            logger.info(f"Ordre placé: {side} {adjusted_quantity} {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Erreur API Binance lors du placement d'ordre {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue placement ordre {symbol}: {e}")
            return None
    
    def manage_position(self, symbol: str, position: Dict):
        """Gère une position existante avec trailing stop et take profit"""
        try:
            df = self.get_klines(symbol, INTERVAL, 50)
            if df.empty:
                return
            
            current_price = df['close'].iloc[-1]
            entry_price = position['entry_price']
            side = position['side']
            quantity = position['quantity']
            
            # Calcul du P&L
            if side == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price
            else:
                pnl_percent = (entry_price - current_price) / entry_price
            
            # Calcul des seuils dynamiques
            dynamic_thresholds = self.calculate_dynamic_volatility_thresholds(df)
            
            # Mise à jour du trailing stop
            if side == 'BUY':
                # Position longue
                if pnl_percent > 0:  # En profit
                    new_stop = current_price * (1 - dynamic_thresholds['stop_loss'])
                    if 'trailing_stop' not in position or new_stop > position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                        logger.info(f"Trailing stop mis à jour pour {symbol}: {new_stop:.6f}")
                
                # Vérification stop loss
                stop_price = position.get('trailing_stop', entry_price * (1 - dynamic_thresholds['stop_loss']))
                if current_price <= stop_price:
                    logger.info(f"Stop loss atteint pour {symbol}: {current_price:.6f} <= {stop_price:.6f}")
                    self.close_position(symbol, 'STOP_LOSS')
                    return
                
                # Vérification take profit
                take_profit_price = entry_price * (1 + dynamic_thresholds['take_profit'])
                if current_price >= take_profit_price:
                    logger.info(f"Take profit atteint pour {symbol}: {current_price:.6f} >= {take_profit_price:.6f}")
                    self.close_position(symbol, 'TAKE_PROFIT')
                    return
            
            else:  # Position courte
                if pnl_percent > 0:  # En profit
                    new_stop = current_price * (1 + dynamic_thresholds['stop_loss'])
                    if 'trailing_stop' not in position or new_stop < position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                        logger.info(f"Trailing stop mis à jour pour {symbol}: {new_stop:.6f}")
                
                # Vérification stop loss
                stop_price = position.get('trailing_stop', entry_price * (1 + dynamic_thresholds['stop_loss']))
                if current_price >= stop_price:
                    logger.info(f"Stop loss atteint pour {symbol}: {current_price:.6f} >= {stop_price:.6f}")
                    self.close_position(symbol, 'STOP_LOSS')
                    return
                
                # Vérification take profit
                take_profit_price = entry_price * (1 - dynamic_thresholds['take_profit'])
                if current_price <= take_profit_price:
                    logger.info(f"Take profit atteint pour {symbol}: {current_price:.6f} <= {take_profit_price:.6f}")
                    self.close_position(symbol, 'TAKE_PROFIT')
                    return
            
            # Mise à jour des statistiques
            position['current_price'] = current_price
            position['pnl_percent'] = pnl_percent
            position['pnl_usdt'] = pnl_percent * position['entry_value']
            
        except Exception as e:
            logger.error(f"Erreur gestion position {symbol}: {e}")
    
    def close_position(self, symbol: str, reason: str = 'MANUAL'):
        """Ferme une position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Pas de position à fermer pour {symbol}")
                return
            
            position = self.positions[symbol]
            side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            quantity = position['quantity']
            
            # Placement de l'ordre de fermeture
            order = self.place_order(symbol, side, quantity)
            
            if order and order.get('status') == 'FILLED':
                # Calcul du résultat final
                exit_price = float(order['fills'][0]['price'])
                entry_price = position['entry_price']
                
                if position['side'] == 'BUY':
                    pnl_percent = (exit_price - entry_price) / entry_price
                else:
                    pnl_percent = (entry_price - exit_price) / entry_price
                
                pnl_usdt = pnl_percent * position['entry_value']
                
                # Enregistrement du trade
                trade_record = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'pnl_percent': pnl_percent,
                    'pnl_usdt': pnl_usdt,
                    'reason': reason
                }
                
                self.trade_history.append(trade_record)
                
                # Notification
                message = f"""
                Position fermée - {reason}
                Symbole: {symbol}
                Côté: {position['side']}
                Prix d'entrée: {entry_price:.6f}
                Prix de sortie: {exit_price:.6f}
                Quantité: {quantity:.6f}
                P&L: {pnl_percent:.2%} ({pnl_usdt:.2f} USDT)
                """
                
                self.send_notification(message, 'CLOSE')
                
                # Suppression de la position
                del self.positions[symbol]
                
                logger.info(f"Position fermée pour {symbol}: {pnl_percent:.2%} ({pnl_usdt:.2f} USDT)")
                
            else:
                logger.error(f"Erreur fermeture position {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur fermeture position {symbol}: {e}")
    
    def check_cooldown(self, symbol: str) -> bool:
        """Vérifie si le cooldown est respecté"""
        if symbol in self.cooldown:
            time_passed = time.time() - self.cooldown[symbol]
            if time_passed < 300:  # 5 minutes de cooldown
                return False
        return True
    
    def should_retrain_ml(self, symbol: str) -> bool:
        """Vérifie si le modèle ML doit être ré-entraîné"""
        if symbol not in self.last_ml_training:
            return True
        
        time_since_training = datetime.now() - self.last_ml_training[symbol]
        return time_since_training > timedelta(hours=ML_RETRAIN_INTERVAL)
    
    def run_strategy(self, symbol: str):
        """Exécute la stratégie pour un symbole"""
        try:
            # Vérification du cooldown
            if not self.check_cooldown(symbol):
                return
            
            # Récupération des données
            df = self.get_klines(symbol, INTERVAL, LOOKBACK_PERIOD)
            if df.empty or len(df) < 100:
                logger.warning(f"Pas assez de données pour {symbol}")
                return
            
            # Calcul des indicateurs
            df = self.calculate_indicators(df)
            
            # Ré-entraînement ML si nécessaire
            if self.should_retrain_ml(symbol):
                self.train_ml_model(symbol)
            
            # Génération du signal
            signal, confidence = self.get_advanced_signal(df, symbol)
            
            # Gestion des positions existantes
            if symbol in self.positions:
                self.manage_position(symbol, self.positions[symbol])
                return
            
            # Vérification du nombre maximum de positions
            if len(self.positions) >= MAX_POSITIONS:
                logger.info(f"Nombre maximum de positions atteint: {MAX_POSITIONS}")
                return
            
            # Ouverture de nouvelles positions
            if signal in ['BUY', 'SELL'] and confidence > 0.6:
                quantity = self.calculate_position_size(symbol, confidence)
                
                if quantity > 0:
                    order = self.place_order(symbol, signal, quantity)
                    
                    if order and order.get('status') == 'FILLED':
                        entry_price = float(order['fills'][0]['price'])
                        entry_value = entry_price * quantity
                        
                        # Enregistrement de la position
                        self.positions[symbol] = {
                            'side': signal,
                            'entry_price': entry_price,
                            'entry_value': entry_value,
                            'quantity': quantity,
                            'entry_time': datetime.now(),
                            'confidence': confidence,
                            'current_price': entry_price,
                            'pnl_percent': 0,
                            'pnl_usdt': 0
                        }
                        
                        # Notification
                        message = f"""
                        Nouvelle position ouverte
                        Symbole: {symbol}
                        Côté: {signal}
                        Prix: {entry_price:.6f}
                        Quantité: {quantity:.6f}
                        Valeur: {entry_value:.2f} USDT
                        Confiance: {confidence:.2%}
                        """
                        
                        self.send_notification(message, signal)
                        
                        # Mise à jour du cooldown
                        self.cooldown[symbol] = time.time()
                        
                        logger.info(f"Position ouverte: {signal} {quantity:.6f} {symbol} à {entry_price:.6f}")
            
            # Mise à jour du dernier signal
            self.last_signals[symbol] = signal
            
        except Exception as e:
            logger.error(f"Erreur stratégie pour {symbol}: {e}")
    
    def get_portfolio_stats(self) -> Dict:
        """Calcule les statistiques du portefeuille"""
        try:
            stats = {
                'total_trades': len(self.trade_history),
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'average_win': 0,
                'average_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'active_positions': len(self.positions),
                'current_positions': {}
            }
            
            if self.trade_history:
                winning_trades = [t for t in self.trade_history if t['pnl_usdt'] > 0]
                losing_trades = [t for t in self.trade_history if t['pnl_usdt'] <= 0]
                
                stats['winning_trades'] = len(winning_trades)
                stats['losing_trades'] = len(losing_trades)
                stats['total_pnl'] = sum(t['pnl_usdt'] for t in self.trade_history)
                stats['win_rate'] = len(winning_trades) / len(self.trade_history) * 100
                
                if winning_trades:
                    stats['average_win'] = sum(t['pnl_usdt'] for t in winning_trades) / len(winning_trades)
                
                if losing_trades:
                    stats['average_loss'] = sum(t['pnl_usdt'] for t in losing_trades) / len(losing_trades)
                
                # Calcul du drawdown
                cumulative_pnl = 0
                peak = 0
                max_drawdown = 0
                
                for trade in self.trade_history:
                    cumulative_pnl += trade['pnl_usdt']
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                stats['max_drawdown'] = max_drawdown * 100
            
            # Positions actuelles
            for symbol, position in self.positions.items():
                stats['current_positions'][symbol] = {
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': position.get('current_price', position['entry_price']),
                    'pnl_percent': position.get('pnl_percent', 0),
                    'pnl_usdt': position.get('pnl_usdt', 0)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
            return {}
    
    def run_backtest(self, start_date: str, end_date: str, initial_balance: float = 10000):
        """Exécute un backtest sur une période donnée"""
        try:
            logger.info(f"Début du backtest du {start_date} au {end_date}")
            
            self.backtest_mode = True
            self.positions = {}
            self.trade_history = []
            
            # Récupération des données historiques pour tous les symboles
            for symbol in SYMBOLS:
                try:
                    # Récupération des données avec plus de lookback pour le backtest
                    df = self.get_klines(symbol, INTERVAL, 1000)
                    if not df.empty:
                        # Filtrage par dates
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        
                        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                        
                        if len(df) > 100:
                            self.backtest_data[symbol] = df
                            logger.info(f"Données backtest chargées pour {symbol}: {len(df)} lignes")
                        else:
                            logger.warning(f"Pas assez de données pour {symbol} sur la période")
                            
                except Exception as e:
                    logger.error(f"Erreur chargement données backtest {symbol}: {e}")
            
            # Entraînement des modèles ML
            for symbol in self.backtest_data.keys():
                self.train_ml_model(symbol)
            
            # Simulation du trading
            total_periods = max(len(df) for df in self.backtest_data.values())
            
            for period in range(100, total_periods):  # Commencer après 100 périodes
                # Mise à jour des données actuelles pour chaque symbole
                for symbol in self.backtest_data.keys():
                    if period < len(self.backtest_data[symbol]):
                        # Limitation des données à la période actuelle
                        current_df = self.backtest_data[symbol].iloc[:period+1]
                        
                        # Sauvegarde temporaire
                        temp_df = self.backtest_data[symbol].copy()
                        self.backtest_data[symbol] = current_df
                        
                        # Exécution de la stratégie
                        self.run_strategy(symbol)
                        
                        # Restauration des données complètes
                        self.backtest_data[symbol] = temp_df
                
                # Affichage du progrès
                if period % 100 == 0:
                    progress = (period / total_periods) * 100
                    logger.info(f"Backtest en cours: {progress:.1f}%")
            
            # Fermeture des positions ouvertes
            for symbol in list(self.positions.keys()):
                self.close_position(symbol, 'BACKTEST_END')
            
            # Calcul des statistiques finales
            stats = self.get_portfolio_stats()
            
            # Rapport de backtest
            logger.info("=== RÉSULTATS DU BACKTEST ===")
            logger.info(f"Période: {start_date} à {end_date}")
            logger.info(f"Nombre total de trades: {stats['total_trades']}")
            logger.info(f"Trades gagnants: {stats['winning_trades']}")
            logger.info(f"Trades perdants: {stats['losing_trades']}")
            logger.info(f"Taux de réussite: {stats['win_rate']:.2f}%")
            logger.info(f"P&L total: {stats['total_pnl']:.2f} USDT")
            logger.info(f"Gain moyen: {stats['average_win']:.2f} USDT")
            logger.info(f"Perte moyenne: {stats['average_loss']:.2f} USDT")
            logger.info(f"Drawdown max: {stats['max_drawdown']:.2f}%")
            
            # Notification des résultats
            backtest_message = f"""
            Backtest terminé ({start_date} à {end_date})
            
            📊 Statistiques:
            • Trades total: {stats['total_trades']}
            • Taux de réussite: {stats['win_rate']:.2f}%
            • P&L total: {stats['total_pnl']:.2f} USDT
            • Drawdown max: {stats['max_drawdown']:.2f}%
            
            ROI: {(stats['total_pnl'] / initial_balance * 100):.2f}%
            """
            
            self.send_notification(backtest_message, 'BACKTEST')
            
            self.backtest_mode = False
            return stats
            
        except Exception as e:
            logger.error(f"Erreur pendant le backtest: {e}")
            self.backtest_mode = False
            return {}
    
    def run(self):
        """Boucle principale du bot"""
        logger.info("Démarrage du bot de trading avancé")
        
        # Entraînement initial des modèles ML
        for symbol in SYMBOLS:
            self.train_ml_model(symbol)
        
        # Notification de démarrage
        self.send_notification("Bot de trading démarré", "START")
        
        try:
            while True:
                start_time = time.time()
                
                # Exécution de la stratégie pour chaque symbole
                for symbol in SYMBOLS:
                    try:
                        self.run_strategy(symbol)
                    except Exception as e:
                        logger.error(f"Erreur stratégie {symbol}: {e}")
                
                # Affichage des statistiques
                stats = self.get_portfolio_stats()
                if stats:
                    logger.info(f"Positions actives: {stats['active_positions']}")
                    logger.info(f"Total P&L: {stats['total_pnl']:.2f} USDT")
                    logger.info(f"Taux de réussite: {stats['win_rate']:.2f}%")
                
                # Temps d'attente
                elapsed = time.time() - start_time
                sleep_time = max(0, SLEEP_TIME - elapsed)
                
                logger.info(f"Cycle terminé en {elapsed:.2f}s, pause de {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Arrêt du bot demandé")
            
            # Fermeture des positions ouvertes
            for symbol in list(self.positions.keys()):
                self.close_position(symbol, 'SHUTDOWN')
            
            # Notification d'arrêt
            self.send_notification("Bot de trading arrêté", "STOP")
            
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            self.send_notification(f"Erreur fatale: {e}", "ERROR")

if __name__ == "__main__":
    # Configuration de base
    bot = AdvancedTradingBot(testnet=True)
    
    # Choix du mode
    mode = input("Choisir le mode (1=Live, 2=Backtest): ")
    if mode == "1":
        # Mode live trading
        try:
            bot.run()
        except KeyboardInterrupt:
            print("Bot arrêté par l'utilisateur")
    elif mode == "2":
        # Mode backtest
        start_date = input("Date de début (YYYY-MM-DD): ")
        end_date = input("Date de fin (YYYY-MM-DD): ")
        initial_balance = float(input("Solde initial (USDT): ") or "10000")
        
        bot.run_backtest(start_date, end_date, initial_balance)
    else:
        print("Mode invalide")

# Ajout des méthodes manquantes dans la classe AdvancedTradingBot
class AdvancedTradingBotContinuation:
    """Méthodes additionnelles pour le bot de trading"""
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Récupère les informations du symbole (minQty, stepSize, etc.)"""
        try:
            if symbol in self.symbol_info_cache:
                return self.symbol_info_cache[symbol]
            
            info = self.client.get_symbol_info(symbol)
            if not info:
                logger.error(f"Impossible de récupérer les infos pour {symbol}")
                return {}
            
            # Extraction des filtres
            filters = {}
            for filter_info in info['filters']:
                filters[filter_info['filterType']] = filter_info
            
            symbol_data = {
                'minQty': float(filters.get('LOT_SIZE', {}).get('minQty', '0.001')),
                'maxQty': float(filters.get('LOT_SIZE', {}).get('maxQty', '1000000')),
                'stepSize': float(filters.get('LOT_SIZE', {}).get('stepSize', '0.001')),
                'minNotional': float(filters.get('MIN_NOTIONAL', {}).get('minNotional', '10')),
                'tickSize': float(filters.get('PRICE_FILTER', {}).get('tickSize', '0.01')),
                'baseAssetPrecision': info['baseAssetPrecision'],
                'quotePrecision': info['quotePrecision']
            }
            
            self.symbol_info_cache[symbol] = symbol_data
            logger.info(f"Infos symbole {symbol}: minQty={symbol_data['minQty']}, stepSize={symbol_data['stepSize']}")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Erreur récupération infos symbole {symbol}: {e}")
            return {}
    
    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        """Ajuste la quantité selon les règles du symbole"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return 0
            
            min_qty = symbol_info['minQty']
            max_qty = symbol_info['maxQty']
            step_size = symbol_info['stepSize']
            
            # Vérification quantité minimum
            if quantity < min_qty:
                logger.warning(f"Quantité {quantity} < minQty {min_qty} pour {symbol}")
                return 0
            
            # Vérification quantité maximum
            if quantity > max_qty:
                quantity = max_qty
                logger.warning(f"Quantité limitée à maxQty {max_qty} pour {symbol}")
            
            # Ajustement selon le stepSize
            if step_size > 0:
                quantity = round(quantity / step_size) * step_size
                quantity = round(quantity, 8)  # Arrondi pour éviter les erreurs de précision
            
            # Vérification finale
            if quantity < min_qty:
                logger.warning(f"Quantité ajustée {quantity} < minQty {min_qty} pour {symbol}")
                return 0
            
            # Vérification du montant minimum (minNotional)
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                notional = quantity * current_price
                
                if notional < symbol_info['minNotional']:
                    # Ajuster la quantité pour respecter le minNotional
                    required_qty = symbol_info['minNotional'] / current_price
                    quantity = max(required_qty, min_qty)
                    
                    # Réajuster selon le stepSize
                    if step_size > 0:
                        quantity = round(quantity / step_size) * step_size
                        quantity = round(quantity, 8)
                    
                    logger.info(f"Quantité ajustée pour minNotional {symbol}: {quantity}")
                    
            except Exception as e:
                logger.error(f"Erreur vérification minNotional {symbol}: {e}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur ajustement quantité {symbol}: {e}")
            return 0
    
    def calculate_dynamic_volatility_thresholds(self, df: pd.DataFrame) -> Dict:
        """Calcule des seuils dynamiques basés sur la volatilité"""
        try:
            if len(df) < 20:
                return {
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'position_size_multiplier': 1.0,
                    'signal_threshold': 0.6
                }
            
            # Calcul de la volatilité récente
            recent_volatility = df['atr_percent'].rolling(20).mean().iloc[-1]
            
            # Ajustement des seuils selon la volatilité
            if recent_volatility < 0.01:  # Très faible volatilité
                thresholds = {
                    'stop_loss': 0.015,
                    'take_profit': 0.025,
                    'position_size_multiplier': 1.2,
                    'signal_threshold': 0.5
                }
            elif recent_volatility < 0.02:  # Faible volatilité
                thresholds = {
                    'stop_loss': 0.02,
                    'take_profit': 0.035,
                    'position_size_multiplier': 1.0,
                    'signal_threshold': 0.6
                }
            elif recent_volatility < 0.04:  # Volatilité normale
                thresholds = {
                    'stop_loss': 0.025,
                    'take_profit': 0.05,
                    'position_size_multiplier': 0.8,
                    'signal_threshold': 0.65
                }
            else:  # Forte volatilité
                thresholds = {
                    'stop_loss': 0.035,
                    'take_profit': 0.07,
                    'position_size_multiplier': 0.6,
                    'signal_threshold': 0.75
                }
            
            logger.debug(f"Seuils dynamiques calculés: volatilité={recent_volatility:.4f}, {thresholds}")
            return thresholds
            
        except Exception as e:
            logger.error(f"Erreur calcul seuils dynamiques: {e}")
            return {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size_multiplier': 1.0,
                'signal_threshold': 0.6
            }
    
    def calculate_multi_timeframe_trend(self, symbol: str) -> Dict:
        """Calcule la tendance sur plusieurs timeframes"""
        try:
            timeframes = ['15m', '1h', '4h', '1d']
            trends = {}
            
            for tf in timeframes:
                try:
                    df = self.get_klines(symbol, tf, 50)
                    if len(df) < 20:
                        continue
                    
                    # Calcul EMA 20 et 50
                    df['ema_20'] = df['close'].ewm(span=20).mean()
                    df['ema_50'] = df['close'].ewm(span=50).mean()
                    
                    # Détermination de la tendance
                    current_price = df['close'].iloc[-1]
                    ema_20 = df['ema_20'].iloc[-1]
                    ema_50 = df['ema_50'].iloc[-1]
                    
                    if current_price > ema_20 > ema_50:
                        trends[tf] = 1  # Tendance haussière
                    elif current_price < ema_20 < ema_50:
                        trends[tf] = -1  # Tendance baissière
                    else:
                        trends[tf] = 0  # Neutre
                    
                except Exception as e:
                    logger.error(f"Erreur calcul tendance {tf} pour {symbol}: {e}")
                    trends[tf] = 0
            
            logger.debug(f"Tendances multi-timeframes pour {symbol}: {trends}")
            return trends
            
        except Exception as e:
            logger.error(f"Erreur calcul tendances multi-timeframes {symbol}: {e}")
            return {}
    
    def get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Obtient une prédiction ML simple"""
        try:
            if symbol not in self.ml_models or len(df) < 100:
                return 0, 0.5
            
            # Préparation des features
            features = self.prepare_ml_features(df)
            if features is None or len(features) == 0:
                return 0, 0.5
            
            # Prédiction
            model = self.ml_models[symbol]
            prediction = model.predict(features.reshape(1, -1))[0]
            
            # Probabilité (si le modèle le supporte)
            try:
                probability = model.predict_proba(features.reshape(1, -1))[0]
                confidence = max(probability)
            except:
                confidence = 0.6  # Confiance par défaut
            
            return int(prediction), confidence
            
        except Exception as e:
            logger.error(f"Erreur prédiction ML {symbol}: {e}")
            return 0, 0.5
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prépare les features pour le ML"""
        try:
            if len(df) < 20:
                return None
            
            # Features techniques
            features = []
            
            # Moyennes mobiles
            features.append(df['ema_fast'].iloc[-1] / df['close'].iloc[-1] - 1)
            features.append(df['ema_slow'].iloc[-1] / df['close'].iloc[-1] - 1)
            features.append(df['ema_trend'].iloc[-1] / df['close'].iloc[-1] - 1)
            
            # RSI normalisé
            features.append((df['rsi'].iloc[-1] - 50) / 50)
            
            # MACD
            features.append(df['macd'].iloc[-1] / df['close'].iloc[-1])
            features.append(df['macd_signal'].iloc[-1] / df['close'].iloc[-1])
            
            # Bollinger Bands
            bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            features.append(bb_position)
            
            # Volatilité
            features.append(df['atr_percent'].iloc[-1])
            
            # Volume relatif
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            features.append(df['volume'].iloc[-1] / avg_volume - 1)
            
            # Momentum
            features.append(df['close'].iloc[-1] / df['close'].iloc[-5] - 1)  # 5 périodes
            features.append(df['close'].iloc[-1] / df['close'].iloc[-20] - 1)  # 20 périodes
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Erreur préparation features ML: {e}")
            return None
    
    def train_ml_model(self, symbol: str):
        """Entraîne un modèle ML simple"""
        try:
            # Récupération des données
            df = self.get_klines(symbol, INTERVAL, 1000)
            if len(df) < 200:
                logger.warning(f"Pas assez de données pour entraîner le modèle {symbol}")
                return
            
            # Calcul des indicateurs
            df = self.calculate_indicators(df)
            
            # Préparation des features
            features_list = []
            targets = []
            
            for i in range(50, len(df) - 10):  # Laisser 10 périodes pour le futur
                features = self.prepare_ml_features(df.iloc[:i+1])
                if features is not None:
                    features_list.append(features)
                    
                    # Target: prix dans 5 périodes > prix actuel
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i + 5]
                    target = 1 if future_price > current_price else 0
                    targets.append(target)
            
            if len(features_list) < 50:
                logger.warning(f"Pas assez de données pour entraîner {symbol}")
                return
            
            # Conversion en arrays numpy
            X = np.array(features_list)
            y = np.array(targets)
            
            # Division train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Entraînement du modèle
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            
            # Test de plusieurs modèles
            models = {
                'rf': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                    
                    logger.info(f"Modèle {name} pour {symbol}: accuracy={score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Erreur entraînement modèle {name}: {e}")
            
            if best_model is not None:
                self.ml_models[symbol] = best_model
                self.last_ml_training[symbol] = datetime.now()
                logger.info(f"Modèle ML entraîné pour {symbol} avec accuracy={best_score:.3f}")
            
        except Exception as e:
            logger.error(f"Erreur entraînement ML {symbol}: {e}")
    
    def send_telegram_notification(self, message: str):
        """Envoie une notification Telegram"""
        try:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                return
                
            import requests
            
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Erreur envoi Telegram: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur notification Telegram: {e}")
    
    def send_discord_notification(self, message: str):
        """Envoie une notification Discord"""
        try:
            if not DISCORD_WEBHOOK_URL:
                return
                
            import requests
            
            data = {
                'content': message,
                'username': 'Trading Bot'
            }
            
            response = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
            if response.status_code != 204:
                logger.error(f"Erreur envoi Discord: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur notification Discord: {e}")
    
    def send_email_notification(self, message: str, subject: str = "Trading Bot Alert"):
        """Envoie une notification par email"""
        try:
            if not EMAIL_CONFIG.get('enabled'):
                return
                
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['from_email']
            msg['To'] = EMAIL_CONFIG['to_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['from_email'], EMAIL_CONFIG['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Erreur notification email: {e}")
    
    def send_notification(self, message: str, trade_type: str = "INFO"):
        """Envoie des notifications sur tous les canaux configurés"""
        try:
            # Formatage du message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp}] {trade_type}\n{message}"
            
            # Envoi sur tous les canaux
            self.send_telegram_notification(formatted_message)
            self.send_discord_notification(formatted_message)
            self.send_email_notification(formatted_message, f"Trading Bot - {trade_type}")
            
        except Exception as e:
            logger.error(f"Erreur envoi notifications: {e}")
    
    def __init__(self, testnet: bool = True):
        """Initialisation du bot avec gestion des erreurs"""
        self.testnet = testnet
        self.positions = {}
        self.trade_history = []
        self.last_signals = {}
        self.cooldown = {}
        self.ml_models = {}
        self.last_ml_training = {}
        self.symbol_info_cache = {}  # Cache pour les infos des symboles
        self.backtest_mode = False
        self.backtest_data = {}
        
        # Configuration du client Binance
        try:
            if testnet:
                self.client = Client(API_KEY, API_SECRET, testnet=True)
                logger.info("Client Binance Testnet initialisé")
            else:
                self.client = Client(API_KEY, API_SECRET)
                logger.info("Client Binance Live initialisé")
                
            # Test de connexion
            self.client.ping()
            
        except Exception as e:
            logger.error(f"Erreur initialisation client Binance: {e}")
            raise
        
        # Pré-chargement des informations des symboles
        self.preload_symbol_info()
    
    def preload_symbol_info(self):
        """Pré-charge les informations des symboles"""
        try:
            logger.info("Pré-chargement des informations des symboles...")
            for symbol in SYMBOLS:
                self.get_symbol_info(symbol)
                time.sleep(0.1)  # Éviter les limites de taux
            logger.info("Informations des symboles chargées")
        except Exception as e:
            logger.error(f"Erreur pré-chargement symboles: {e}")

# Configuration des notifications
TELEGRAM_BOT_TOKEN = ""  # Token du bot Telegram
TELEGRAM_CHAT_ID = ""    # ID du chat Telegram

DISCORD_WEBHOOK_URL = ""  # URL du webhook Discord

EMAIL_CONFIG = {
    'enabled': False,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'from_email': '',
    'password': '',  # Mot de passe d'application
    'to_email': ''
}

# Mise à jour de la classe principale
def update_trading_bot_class():
    """Met à jour la classe AdvancedTradingBot avec les nouvelles méthodes"""
    
    # Ajout des méthodes de la continuation à la classe principale
    for method_name in dir(AdvancedTradingBotContinuation):
        if not method_name.startswith('_'):
            method = getattr(AdvancedTradingBotContinuation, method_name)
            if callable(method):
                setattr(AdvancedTradingBot, method_name, method)

# Application des mises à jour
update_trading_bot_class()

# Fonction principale améliorée
def main():
    """Fonction principale avec gestion d'erreurs"""
    try:
        print("=== BOT DE TRADING AVANCÉ ===")
        print("1. Live Trading")
        print("2. Backtest")
        print("3. Test des notifications")
        print("4. Analyse des symboles")
        
        choice = input("Choisir une option (1-4): ")
        
        if choice == "1":
            # Live trading
            testnet = input("Utiliser le testnet? (y/n): ").lower() == 'y'
            bot = AdvancedTradingBot(testnet=testnet)
            
            print("Démarrage du trading en direct...")
            bot.run()
            
        elif choice == "2":
            # Backtest
            bot = AdvancedTradingBot(testnet=True)
            
            start_date = input("Date de début (YYYY-MM-DD): ")
            end_date = input("Date de fin (YYYY-MM-DD): ")
            initial_balance = float(input("Solde initial (défaut: 10000): ") or "10000")
            
            print("Démarrage du backtest...")
            results = bot.run_backtest(start_date, end_date, initial_balance)
            
            if results:
                print("\n=== RÉSULTATS DU BACKTEST ===")
                print(f"Trades total: {results['total_trades']}")
                print(f"Taux de réussite: {results['win_rate']:.2f}%")
                print(f"P&L total: {results['total_pnl']:.2f} USDT")
                print(f"ROI: {(results['total_pnl'] / initial_balance * 100):.2f}%")
                
        elif choice == "3":
            # Test des notifications
            bot = AdvancedTradingBot(testnet=True)
            test_message = "Test de notification du bot de trading"
            
            print("Envoi des notifications de test...")
            bot.send_notification(test_message, "TEST")
            print("Notifications envoyées!")
            
        elif choice == "4":
            # Analyse des symboles
            bot = AdvancedTradingBot(testnet=True)
            
            print("Analyse des symboles en cours...")
            for symbol in SYMBOLS:
                try:
                    info = bot.get_symbol_info(symbol)
                    if info:
                        print(f"{symbol}: minQty={info['minQty']}, minNotional={info['minNotional']}")
                except Exception as e:
                    print(f"Erreur pour {symbol}: {e}")
                    
        else:
            print("Option invalide")
            
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur dans main: {e}")
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()