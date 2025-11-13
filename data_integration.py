import asyncio
import aiohttp
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import time


class DataSource(Enum):
    PBOC_CREDIT = "pboc_credit"
    ZHIMA_CREDIT = "zhima_credit"
    WECHAT_SCORE = "wechat_score"
    BANK_INTERNAL = "bank_internal"
    THIRD_PARTY_CREDIT = "third_party_credit"


@dataclass
class APICredentials:
    api_key: str
    api_secret: str
    endpoint: str
    auth_type: str
    rate_limit: int
    timeout: int


@dataclass
class DataIntegrationRequest:
    user_id: str
    data_source: DataSource
    request_type: str
    parameters: Dict[str, Any]
    encryption_required: bool = True
    cache_duration: int = 3600


@dataclass
class IntegrationResponse:
    success: bool
    data: Optional[Dict]
    error: Optional[str]
    timestamp: datetime
    source: DataSource
    cache_hit: bool
    processing_time: float


class DataEncryption:
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self.cipher_suite = None
        self._initialize_cipher()
    
    def _initialize_cipher(self):
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'credit_health_salt_2024',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher_suite = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        encrypted = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        decoded = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher_suite.decrypt(decoded)
        return decrypted.decode()
    
    def hash_sensitive_field(self, field_value: str) -> str:
        return hashlib.sha256(f"{field_value}_{self.master_key}".encode()).hexdigest()


class BaseDataConnector(ABC):
    def __init__(self, credentials: APICredentials, encryption: DataEncryption):
        self.credentials = credentials
        self.encryption = encryption
        self.session = None
        self.rate_limiter = RateLimiter(credentials.rate_limit)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch_data(self, request: DataIntegrationRequest) -> Dict:
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: Dict) -> Dict:
        pass
    
    async def make_request(self, endpoint: str, method: str, 
                          data: Optional[Dict] = None) -> Dict:
        await self.rate_limiter.wait_if_needed()
        
        headers = self._generate_auth_headers()
        
        url = f"{self.credentials.endpoint}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(
                    url, 
                    headers=headers, 
                    timeout=self.credentials.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            
            elif method.upper() == 'POST':
                async with self.session.post(
                    url, 
                    headers=headers, 
                    json=data,
                    timeout=self.credentials.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        
        except aiohttp.ClientError as e:
            raise ConnectionError(f"API request failed: {str(e)}")
    
    def _generate_auth_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        
        if self.credentials.auth_type == 'hmac':
            signature = self._generate_hmac_signature(timestamp)
            return {
                'X-API-Key': self.credentials.api_key,
                'X-Timestamp': timestamp,
                'X-Signature': signature,
                'Content-Type': 'application/json'
            }
        
        elif self.credentials.auth_type == 'bearer':
            return {
                'Authorization': f'Bearer {self.credentials.api_key}',
                'Content-Type': 'application/json'
            }
        
        else:
            return {
                'X-API-Key': self.credentials.api_key,
                'Content-Type': 'application/json'
            }
    
    def _generate_hmac_signature(self, timestamp: str) -> str:
        message = f"{self.credentials.api_key}{timestamp}".encode()
        signature = hmac.new(
            self.credentials.api_secret.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature


class PBOCCreditConnector(BaseDataConnector):
    async def fetch_data(self, request: DataIntegrationRequest) -> Dict:
        endpoint = '/api/v2/credit/report'
        
        request_data = {
            'user_id': self.encryption.hash_sensitive_field(request.user_id),
            'report_type': request.parameters.get('report_type', 'full'),
            'include_details': request.parameters.get('include_details', True)
        }
        
        response = await self.make_request(endpoint, 'POST', request_data)
        
        return response
    
    def transform_data(self, raw_data: Dict) -> Dict:
        transformed = {
            'credit_score': raw_data.get('score', 0),
            'credit_history': {
                'length_months': raw_data.get('credit_history_length', 0),
                'oldest_account': raw_data.get('oldest_account_date'),
                'newest_account': raw_data.get('newest_account_date')
            },
            'accounts': {
                'total': raw_data.get('total_accounts', 0),
                'active': raw_data.get('active_accounts', 0),
                'closed': raw_data.get('closed_accounts', 0)
            },
            'credit_utilization': {
                'total_limit': raw_data.get('total_credit_limit', 0),
                'total_used': raw_data.get('total_credit_used', 0),
                'utilization_ratio': raw_data.get('utilization_ratio', 0)
            },
            'payment_history': {
                'on_time_payments': raw_data.get('on_time_count', 0),
                'late_payments': raw_data.get('late_payment_count', 0),
                'delinquencies': raw_data.get('delinquencies', [])
            },
            'inquiries': {
                'hard_inquiries_6m': raw_data.get('hard_inquiries_6m', 0),
                'hard_inquiries_12m': raw_data.get('hard_inquiries_12m', 0),
                'soft_inquiries': raw_data.get('soft_inquiries', 0)
            },
            'debt_details': {
                'total_debt': raw_data.get('total_debt', 0),
                'mortgage_debt': raw_data.get('mortgage_debt', 0),
                'auto_loan_debt': raw_data.get('auto_loan_debt', 0),
                'credit_card_debt': raw_data.get('credit_card_debt', 0),
                'student_loan_debt': raw_data.get('student_loan_debt', 0)
            },
            'public_records': {
                'bankruptcies': raw_data.get('bankruptcies', 0),
                'tax_liens': raw_data.get('tax_liens', 0),
                'judgments': raw_data.get('judgments', 0)
            }
        }
        
        return transformed


class ZhimaCreditConnector(BaseDataConnector):
    async def fetch_data(self, request: DataIntegrationRequest) -> Dict:
        endpoint = '/openapi/score/get'
        
        request_data = {
            'transaction_id': self._generate_transaction_id(request.user_id),
            'product_code': 'credit_score',
            'biz_params': json.dumps({
                'user_id': request.user_id,
                'scene': 'credit_check'
            })
        }
        
        response = await self.make_request(endpoint, 'POST', request_data)
        
        return response
    
    def transform_data(self, raw_data: Dict) -> Dict:
        score_data = raw_data.get('data', {})
        
        transformed = {
            'zhima_score': score_data.get('zm_score', 0),
            'score_level': self._convert_score_to_level(score_data.get('zm_score', 0)),
            'update_time': score_data.get('score_time'),
            'dimensions': {
                'credit_history': score_data.get('credit_history_score', 0),
                'behavior_preference': score_data.get('behavior_score', 0),
                'performance_capacity': score_data.get('performance_score', 0),
                'identity_verification': score_data.get('identity_score', 0),
                'connections': score_data.get('connection_score', 0)
            }
        }
        
        return transformed
    
    def _generate_transaction_id(self, user_id: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"TXN{timestamp}{user_id[:8]}"
    
    def _convert_score_to_level(self, score: int) -> str:
        if score >= 750:
            return 'excellent'
        elif score >= 700:
            return 'very_good'
        elif score >= 650:
            return 'good'
        elif score >= 600:
            return 'fair'
        else:
            return 'poor'


class WeChatScoreConnector(BaseDataConnector):
    async def fetch_data(self, request: DataIntegrationRequest) -> Dict:
        endpoint = '/cgi-bin/creditapi/score/query'
        
        request_data = {
            'openid': request.parameters.get('wechat_openid'),
            'appid': self.credentials.api_key,
            'nonce_str': self._generate_nonce(),
            'timestamp': int(time.time())
        }
        
        request_data['sign'] = self._generate_wechat_signature(request_data)
        
        response = await self.make_request(endpoint, 'POST', request_data)
        
        return response
    
    def transform_data(self, raw_data: Dict) -> Dict:
        transformed = {
            'wechat_score': raw_data.get('credit_score', 0),
            'score_grade': raw_data.get('grade'),
            'score_range': {
                'min': raw_data.get('score_range', {}).get('min', 0),
                'max': raw_data.get('score_range', {}).get('max', 1000)
            },
            'factors': {
                'payment_behavior': raw_data.get('payment_factor', 0),
                'social_connections': raw_data.get('social_factor', 0),
                'consumption_pattern': raw_data.get('consumption_factor', 0),
                'account_info': raw_data.get('account_factor', 0)
            }
        }
        
        return transformed
    
    def _generate_nonce(self) -> str:
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    def _generate_wechat_signature(self, params: Dict) -> str:
        sorted_params = sorted(params.items())
        sign_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
        sign_str += f"&key={self.credentials.api_secret}"
        
        return hashlib.md5(sign_str.encode()).hexdigest().upper()


class BankInternalConnector(BaseDataConnector):
    async def fetch_data(self, request: DataIntegrationRequest) -> Dict:
        endpoint = '/internal/api/customer/credit'
        
        request_data = {
            'customer_id': request.user_id,
            'data_types': request.parameters.get('data_types', [
                'transactions', 'accounts', 'loans', 'investments'
            ]),
            'time_range': request.parameters.get('time_range', 'last_12_months')
        }
        
        response = await self.make_request(endpoint, 'POST', request_data)
        
        return response
    
    def transform_data(self, raw_data: Dict) -> Dict:
        transformed = {
            'account_info': {
                'account_age_months': raw_data.get('account_age_months', 0),
                'account_types': raw_data.get('account_types', []),
                'average_balance': raw_data.get('avg_balance', 0)
            },
            'transaction_behavior': {
                'monthly_income': raw_data.get('monthly_income_avg', 0),
                'monthly_expenses': raw_data.get('monthly_expenses_avg', 0),
                'savings_rate': raw_data.get('savings_rate', 0),
                'transaction_frequency': raw_data.get('transaction_freq', 0)
            },
            'loan_info': {
                'active_loans': raw_data.get('active_loans', []),
                'total_loan_amount': raw_data.get('total_loan_amount', 0),
                'monthly_payment': raw_data.get('monthly_payment', 0),
                'payment_history': raw_data.get('payment_history', [])
            },
            'investment_profile': {
                'total_investment': raw_data.get('total_investment', 0),
                'investment_types': raw_data.get('investment_types', []),
                'risk_preference': raw_data.get('risk_preference', 'moderate')
            },
            'banking_relationship': {
                'relationship_length': raw_data.get('relationship_months', 0),
                'product_count': raw_data.get('product_count', 0),
                'service_usage_score': raw_data.get('service_score', 0)
            }
        }
        
        return transformed


class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.window_size = 60
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.window_size]
        
        if len(self.requests) >= self.max_requests:
            wait_time = self.window_size - (current_time - self.requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.requests = []
        
        self.requests.append(current_time)


class DataIntegrationService:
    def __init__(self, master_encryption_key: str, redis_client: redis.Redis):
        self.encryption = DataEncryption(master_encryption_key)
        self.redis_client = redis_client
        self.connectors = {}
        self.response_cache = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        self.connector_configs = {
            DataSource.PBOC_CREDIT: {
                'class': PBOCCreditConnector,
                'credentials': APICredentials(
                    api_key='pboc_api_key',
                    api_secret='pboc_api_secret',
                    endpoint='https://api.pboc.credit.cn',
                    auth_type='hmac',
                    rate_limit=30,
                    timeout=30
                )
            },
            DataSource.ZHIMA_CREDIT: {
                'class': ZhimaCreditConnector,
                'credentials': APICredentials(
                    api_key='zhima_app_id',
                    api_secret='zhima_app_secret',
                    endpoint='https://openapi.alipay.com',
                    auth_type='hmac',
                    rate_limit=60,
                    timeout=15
                )
            },
            DataSource.WECHAT_SCORE: {
                'class': WeChatScoreConnector,
                'credentials': APICredentials(
                    api_key='wechat_app_id',
                    api_secret='wechat_app_secret',
                    endpoint='https://api.weixin.qq.com',
                    auth_type='hmac',
                    rate_limit=50,
                    timeout=20
                )
            },
            DataSource.BANK_INTERNAL: {
                'class': BankInternalConnector,
                'credentials': APICredentials(
                    api_key='internal_api_key',
                    api_secret='internal_api_secret',
                    endpoint='https://internal.bank.api',
                    auth_type='bearer',
                    rate_limit=100,
                    timeout=10
                )
            }
        }
    
    async def fetch_integrated_data(self, request: DataIntegrationRequest) -> IntegrationResponse:
        start_time = time.time()
        
        cache_key = self._generate_cache_key(request)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data:
            return IntegrationResponse(
                success=True,
                data=cached_data,
                error=None,
                timestamp=datetime.now(),
                source=request.data_source,
                cache_hit=True,
                processing_time=time.time() - start_time
            )
        
        try:
            connector_config = self.connector_configs.get(request.data_source)
            
            if not connector_config:
                raise ValueError(f"Unsupported data source: {request.data_source}")
            
            connector_class = connector_config['class']
            credentials = connector_config['credentials']
            
            async with connector_class(credentials, self.encryption) as connector:
                raw_data = await connector.fetch_data(request)
                
                transformed_data = connector.transform_data(raw_data)
                
                if request.encryption_required:
                    sensitive_fields = ['user_id', 'id_number', 'phone_number', 'account_number']
                    transformed_data = self._encrypt_sensitive_fields(
                        transformed_data, sensitive_fields
                    )
                
                self._store_in_cache(cache_key, transformed_data, request.cache_duration)
                
                return IntegrationResponse(
                    success=True,
                    data=transformed_data,
                    error=None,
                    timestamp=datetime.now(),
                    source=request.data_source,
                    cache_hit=False,
                    processing_time=time.time() - start_time
                )
        
        except Exception as e:
            return IntegrationResponse(
                success=False,
                data=None,
                error=str(e),
                timestamp=datetime.now(),
                source=request.data_source,
                cache_hit=False,
                processing_time=time.time() - start_time
            )
    
    async def fetch_multi_source_data(self, user_id: str, 
                                     sources: List[DataSource]) -> Dict[DataSource, IntegrationResponse]:
        tasks = []
        
        for source in sources:
            request = DataIntegrationRequest(
                user_id=user_id,
                data_source=source,
                request_type='full_report',
                parameters={},
                encryption_required=True,
                cache_duration=3600
            )
            tasks.append(self.fetch_integrated_data(request))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for source, response in zip(sources, responses):
            if isinstance(response, Exception):
                result[source] = IntegrationResponse(
                    success=False,
                    data=None,
                    error=str(response),
                    timestamp=datetime.now(),
                    source=source,
                    cache_hit=False,
                    processing_time=0
                )
            else:
                result[source] = response
        
        return result
    
    def _generate_cache_key(self, request: DataIntegrationRequest) -> str:
        key_components = [
            request.user_id,
            request.data_source.value,
            request.request_type,
            json.dumps(request.parameters, sort_keys=True)
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        try:
            cached_value = self.redis_client.get(cache_key)
            if cached_value:
                return json.loads(cached_value)
        except Exception:
            pass
        return None
    
    def _store_in_cache(self, cache_key: str, data: Dict, duration: int):
        try:
            self.redis_client.setex(
                cache_key,
                duration,
                json.dumps(data)
            )
        except Exception:
            pass
    
    def _encrypt_sensitive_fields(self, data: Dict, sensitive_fields: List[str]) -> Dict:
        encrypted_data = data.copy()
        
        def encrypt_recursive(obj, fields):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in fields and isinstance(value, str):
                        obj[key] = self.encryption.encrypt(value)
                    elif isinstance(value, (dict, list)):
                        encrypt_recursive(value, fields)
            elif isinstance(obj, list):
                for item in obj:
                    encrypt_recursive(item, fields)
        
        encrypt_recursive(encrypted_data, sensitive_fields)
        return encrypted_data
    
    async def validate_user_authorization(self, user_id: str, 
                                         data_source: DataSource) -> bool:
        auth_key = f"auth:{user_id}:{data_source.value}"
        
        try:
            auth_status = self.redis_client.get(auth_key)
            if auth_status:
                auth_data = json.loads(auth_status)
                expiry = datetime.fromisoformat(auth_data['expiry'])
                return expiry > datetime.now()
        except Exception:
            pass
        
        return False
    
    async def store_user_authorization(self, user_id: str, 
                                      data_source: DataSource, 
                                      duration_days: int = 90):
        auth_key = f"auth:{user_id}:{data_source.value}"
        
        auth_data = {
            'user_id': user_id,
            'data_source': data_source.value,
            'authorized_at': datetime.now().isoformat(),
            'expiry': (datetime.now() + timedelta(days=duration_days)).isoformat()
        }
        
        self.redis_client.setex(
            auth_key,
            duration_days * 86400,
            json.dumps(auth_data)
        )
    
    async def revoke_user_authorization(self, user_id: str, data_source: DataSource):
        auth_key = f"auth:{user_id}:{data_source.value}"
        self.redis_client.delete(auth_key)