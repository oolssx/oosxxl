import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from enum import Enum
import hashlib
from collections import defaultdict, deque


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    LOAN_APPLICATION = "loan_application"
    CREDIT_INQUIRY = "credit_inquiry"
    CREDIT_CARD_APPLICATION = "credit_card_application"
    LARGE_PURCHASE = "large_purchase"
    ACCOUNT_CLOSURE = "account_closure"
    PAYMENT_MISSED = "payment_missed"
    HIGH_UTILIZATION = "high_utilization"
    MULTIPLE_INQUIRIES = "multiple_inquiries"


class CreditWarningEngine:
    def __init__(self, predictor_model, notification_service, database_service):
        self.predictor = predictor_model
        self.notification_service = notification_service
        self.database = database_service
        self.risk_rules = self._initialize_risk_rules()
        self.user_action_history = defaultdict(lambda: deque(maxlen=100))
        self.warning_cache = {}
        self.simulation_cache = {}
        self.monitoring_active = True
        
    def _initialize_risk_rules(self):
        return {
            ActionType.LOAN_APPLICATION: {
                'weight': 0.8,
                'threshold': 0.4,
                'cooldown_days': 30,
                'impact_factors': {
                    'inquiry_count': 0.3,
                    'existing_debt': 0.25,
                    'recent_applications': 0.25,
                    'credit_score': 0.2
                }
            },
            ActionType.CREDIT_INQUIRY: {
                'weight': 0.5,
                'threshold': 0.3,
                'cooldown_days': 7,
                'impact_factors': {
                    'recent_inquiries': 0.4,
                    'credit_score': 0.3,
                    'account_age': 0.3
                }
            },
            ActionType.HIGH_UTILIZATION: {
                'weight': 0.7,
                'threshold': 0.5,
                'cooldown_days': 14,
                'impact_factors': {
                    'utilization_ratio': 0.4,
                    'payment_history': 0.3,
                    'available_credit': 0.3
                }
            },
            ActionType.PAYMENT_MISSED: {
                'weight': 0.95,
                'threshold': 0.2,
                'cooldown_days': 60,
                'impact_factors': {
                    'days_overdue': 0.4,
                    'amount': 0.3,
                    'history': 0.3
                }
            },
            ActionType.MULTIPLE_INQUIRIES: {
                'weight': 0.75,
                'threshold': 0.35,
                'cooldown_days': 21,
                'impact_factors': {
                    'inquiry_velocity': 0.5,
                    'inquiry_count': 0.3,
                    'credit_score': 0.2
                }
            }
        }
    
    async def monitor_user_action(self, user_id: str, action_type: ActionType, 
                                  action_data: Dict[str, Any]) -> Optional[Dict]:
        try:
            if not self.monitoring_active:
                return None
            
            action_timestamp = datetime.now()
            
            self.user_action_history[user_id].append({
                'type': action_type,
                'data': action_data,
                'timestamp': action_timestamp
            })
            
            current_state = await self._get_user_credit_state(user_id)
            
            if not current_state:
                return None
            
            risk_assessment = await self._assess_risk(
                user_id, current_state, action_type, action_data
            )
            
            if risk_assessment['risk_level'] == RiskLevel.LOW:
                await self._log_action(user_id, action_type, risk_assessment, False)
                return None
            
            impact_simulation = await self._simulate_impact(
                user_id, current_state, action_type, action_data
            )
            
            warning_content = await self._generate_warning_content(
                user_id, current_state, action_type, action_data,
                risk_assessment, impact_simulation
            )
            
            warning_message = {
                'user_id': user_id,
                'timestamp': action_timestamp.isoformat(),
                'action_type': action_type.value,
                'risk_level': risk_assessment['risk_level'].value,
                'risk_score': risk_assessment['risk_score'],
                'impact_prediction': impact_simulation,
                'warning_content': warning_content,
                'should_block': risk_assessment['risk_level'] == RiskLevel.CRITICAL
            }
            
            await self._send_warning(user_id, warning_message)
            
            await self._log_action(user_id, action_type, risk_assessment, True)
            
            return warning_message
            
        except Exception as e:
            print(f"Error monitoring action for user {user_id}: {str(e)}")
            return None
    
    async def _get_user_credit_state(self, user_id: str) -> Optional[Dict]:
        cache_key = f"credit_state_{user_id}"
        
        if cache_key in self.warning_cache:
            cached_data = self.warning_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 3600:
                return cached_data['data']
        
        user_data = await self.database.get_user_credit_data(user_id)
        
        if not user_data:
            return None
        
        user_history = await self.database.get_user_history(user_id)
        
        current_score_prediction = self.predictor.predict(
            user_data['features'],
            user_history
        )
        
        credit_state = {
            'user_id': user_id,
            'current_score': current_score_prediction['score'],
            'score_confidence': current_score_prediction['confidence'],
            'features': user_data['features'],
            'history': user_history,
            'recent_actions': list(self.user_action_history[user_id])[-10:],
            'credit_utilization': user_data['features'].get('credit_utilization_ratio', 0),
            'total_debt': user_data['features'].get('total_debt', 0),
            'payment_history_score': user_data['features'].get('payment_history_score', 0),
            'hard_inquiries_6m': user_data['features'].get('hard_inquiries_6m', 0),
            'last_updated': datetime.now()
        }
        
        self.warning_cache[cache_key] = {
            'data': credit_state,
            'timestamp': datetime.now()
        }
        
        return credit_state
    
    async def _assess_risk(self, user_id: str, current_state: Dict, 
                          action_type: ActionType, action_data: Dict) -> Dict:
        
        if action_type not in self.risk_rules:
            return {
                'risk_level': RiskLevel.LOW,
                'risk_score': 0.1,
                'factors': {}
            }
        
        rule = self.risk_rules[action_type]
        
        recent_same_actions = [
            a for a in self.user_action_history[user_id]
            if a['type'] == action_type
            and (datetime.now() - a['timestamp']).days < rule['cooldown_days']
        ]
        
        if len(recent_same_actions) >= 3:
            frequency_penalty = 0.3
        elif len(recent_same_actions) >= 1:
            frequency_penalty = 0.15
        else:
            frequency_penalty = 0
        
        factor_scores = {}
        for factor, weight in rule['impact_factors'].items():
            factor_scores[factor] = self._evaluate_risk_factor(
                factor, current_state, action_data
            )
        
        weighted_risk = sum(
            score * rule['impact_factors'][factor]
            for factor, score in factor_scores.items()
        )
        
        base_risk = weighted_risk * rule['weight']
        
        total_risk = min(1.0, base_risk + frequency_penalty)
        
        if total_risk >= 0.75:
            risk_level = RiskLevel.CRITICAL
        elif total_risk >= 0.55:
            risk_level = RiskLevel.HIGH
        elif total_risk >= 0.35:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'risk_level': risk_level,
            'risk_score': total_risk,
            'base_risk': base_risk,
            'frequency_penalty': frequency_penalty,
            'factor_scores': factor_scores,
            'recent_actions_count': len(recent_same_actions)
        }
    
    def _evaluate_risk_factor(self, factor: str, current_state: Dict, 
                             action_data: Dict) -> float:
        
        if factor == 'inquiry_count':
            count = current_state.get('hard_inquiries_6m', 0)
            return min(1.0, count / 6.0)
        
        elif factor == 'existing_debt':
            debt = current_state.get('total_debt', 0)
            income = action_data.get('monthly_income', 5000)
            dti = debt / (income * 12) if income > 0 else 1.0
            return min(1.0, dti / 0.5)
        
        elif factor == 'recent_applications':
            recent_count = len([
                a for a in current_state.get('recent_actions', [])
                if a['type'] == ActionType.LOAN_APPLICATION
            ])
            return min(1.0, recent_count / 3.0)
        
        elif factor == 'credit_score':
            score = current_state.get('current_score', 700)
            return max(0, (750 - score) / 250)
        
        elif factor == 'recent_inquiries':
            count = current_state.get('hard_inquiries_6m', 0)
            return min(1.0, count / 5.0)
        
        elif factor == 'utilization_ratio':
            ratio = current_state.get('credit_utilization', 0)
            return min(1.0, max(0, (ratio - 0.3) / 0.5))
        
        elif factor == 'payment_history':
            history_score = current_state.get('payment_history_score', 100)
            return max(0, (100 - history_score) / 100)
        
        elif factor == 'inquiry_velocity':
            recent_inquiries = [
                a for a in current_state.get('recent_actions', [])
                if a['type'] == ActionType.CREDIT_INQUIRY
                and (datetime.now() - a['timestamp']).days < 30
            ]
            return min(1.0, len(recent_inquiries) / 4.0)
        
        return 0.5
    
    async def _simulate_impact(self, user_id: str, current_state: Dict,
                               action_type: ActionType, action_data: Dict) -> Dict:
        
        cache_key = hashlib.md5(
            f"{user_id}_{action_type.value}_{json.dumps(action_data)}".encode()
        ).hexdigest()
        
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        changes = self._calculate_feature_changes(action_type, action_data, current_state)
        
        simulation_result = self.predictor.simulate_scenario(
            current_state['features'],
            current_state['history'],
            changes
        )
        
        time_horizons = {
            '1_month': self._project_score_recovery(simulation_result['score_change'], 1),
            '3_months': self._project_score_recovery(simulation_result['score_change'], 3),
            '6_months': self._project_score_recovery(simulation_result['score_change'], 6),
            '12_months': self._project_score_recovery(simulation_result['score_change'], 12)
        }
        
        impact_result = {
            'immediate_impact': simulation_result['score_change'],
            'current_score': simulation_result['original_score'],
            'projected_score': simulation_result['new_score'],
            'confidence': simulation_result['new_confidence'],
            'time_projections': time_horizons,
            'feature_changes': changes,
            'recovery_time_estimate': self._estimate_recovery_time(
                simulation_result['score_change']
            )
        }
        
        self.simulation_cache[cache_key] = impact_result
        
        return impact_result
    
    def _calculate_feature_changes(self, action_type: ActionType, 
                                   action_data: Dict, current_state: Dict) -> Dict:
        changes = {}
        
        if action_type == ActionType.LOAN_APPLICATION:
            changes['hard_inquiries_6m'] = {
                'type': 'relative',
                'value': 0.2
            }
            if 'loan_amount' in action_data:
                changes['total_debt'] = {
                    'type': 'absolute',
                    'value': current_state['total_debt'] + action_data['loan_amount']
                }
        
        elif action_type == ActionType.CREDIT_INQUIRY:
            changes['hard_inquiries_6m'] = {
                'type': 'relative',
                'value': 0.15
            }
            changes['inquiries_last_month'] = {
                'type': 'relative',
                'value': 0.25
            }
        
        elif action_type == ActionType.HIGH_UTILIZATION:
            if 'new_utilization' in action_data:
                changes['credit_utilization_ratio'] = {
                    'type': 'absolute',
                    'value': action_data['new_utilization']
                }
        
        elif action_type == ActionType.PAYMENT_MISSED:
            changes['total_overdue_count'] = {
                'type': 'relative',
                'value': 0.3
            }
            changes['payment_history_score'] = {
                'type': 'relative',
                'value': -0.15
            }
        
        return changes
    
    def _project_score_recovery(self, initial_impact: float, months: int) -> float:
        decay_rate = 0.15
        
        recovery = initial_impact * np.exp(-decay_rate * months)
        
        return float(recovery)
    
    def _estimate_recovery_time(self, score_impact: float) -> int:
        if score_impact >= -5:
            return 1
        elif score_impact >= -15:
            return 3
        elif score_impact >= -30:
            return 6
        elif score_impact >= -50:
            return 12
        else:
            return 18
    
    async def _generate_warning_content(self, user_id: str, current_state: Dict,
                                       action_type: ActionType, action_data: Dict,
                                       risk_assessment: Dict, impact_simulation: Dict) -> Dict:
        
        user_profile = await self.database.get_user_profile(user_id)
        user_plans = await self.database.get_user_future_plans(user_id)
        
        content = {
            'title': self._generate_warning_title(action_type, risk_assessment),
            'severity': risk_assessment['risk_level'].value,
            'summary': self._generate_summary(action_type, impact_simulation),
            'detailed_analysis': self._generate_detailed_analysis(
                risk_assessment, impact_simulation
            ),
            'recommendations': self._generate_recommendations(
                action_type, current_state, impact_simulation, user_plans
            ),
            'alternatives': self._generate_alternatives(
                action_type, action_data, current_state
            ),
            'visualization_data': self._prepare_visualization_data(
                current_state, impact_simulation
            )
        }
        
        return content
    
    def _generate_warning_title(self, action_type: ActionType, 
                                risk_assessment: Dict) -> str:
        titles = {
            RiskLevel.CRITICAL: "⚠️ 严重信用风险警告",
            RiskLevel.HIGH: "🔴 高风险提醒",
            RiskLevel.MEDIUM: "🟡 中度风险提示",
            RiskLevel.LOW: "🟢 友情提醒"
        }
        return titles.get(risk_assessment['risk_level'], "提醒")
    
    def _generate_summary(self, action_type: ActionType, 
                         impact_simulation: Dict) -> str:
        score_change = impact_simulation['immediate_impact']
        
        if score_change < -20:
            severity = "显著降低"
        elif score_change < -10:
            severity = "降低"
        elif score_change < -5:
            severity = "轻微降低"
        else:
            severity = "可能影响"
        
        return f"本次操作预计会{severity}您的信用评分约{abs(score_change):.0f}分"
    
    def _generate_detailed_analysis(self, risk_assessment: Dict, 
                                    impact_simulation: Dict) -> List[str]:
        analysis = []
        
        analysis.append(f"当前信用评分：{impact_simulation['current_score']:.0f}分")
        analysis.append(f"预计变化后评分：{impact_simulation['projected_score']:.0f}分")
        
        if risk_assessment['frequency_penalty'] > 0:
            analysis.append(f"检测到您最近频繁进行类似操作，额外风险增加")
        
        top_factors = sorted(
            risk_assessment['factor_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for factor, score in top_factors:
            if score > 0.5:
                analysis.append(f"风险因素：{factor}（风险度：{score:.0%}）")
        
        return analysis
    
    def _generate_recommendations(self, action_type: ActionType, 
                                 current_state: Dict, impact_simulation: Dict,
                                 user_plans: Optional[Dict]) -> List[Dict]:
        recommendations = []
        
        if action_type == ActionType.LOAN_APPLICATION:
            if user_plans and 'mortgage' in user_plans:
                time_to_plan = (user_plans['mortgage']['target_date'] - datetime.now()).days
                if time_to_plan < 180:
                    recommendations.append({
                        'priority': 'high',
                        'text': f"您计划在{time_to_plan}天后申请房贷，建议推迟本次贷款申请",
                        'reason': "频繁的贷款申请可能降低房贷审批通过率"
                    })
            
            if current_state['hard_inquiries_6m'] >= 4:
                recommendations.append({
                    'priority': 'high',
                    'text': "您近6个月已有多次信用查询记录，建议暂缓申请",
                    'reason': "查询次数过多可能被视为资金紧张信号"
                })
        
        if impact_simulation['immediate_impact'] < -15:
            recommendations.append({
                'priority': 'medium',
                'text': f"预计需要{impact_simulation['recovery_time_estimate']}个月才能恢复",
                'reason': "建议评估是否有其他替代方案"
            })
        
        return recommendations
    
    def _generate_alternatives(self, action_type: ActionType, 
                              action_data: Dict, current_state: Dict) -> List[Dict]:
        alternatives = []
        
        if action_type == ActionType.LOAN_APPLICATION:
            if current_state.get('available_credit', 0) > action_data.get('loan_amount', 0):
                alternatives.append({
                    'type': '信用卡分期',
                    'description': '使用现有信用卡额度进行分期',
                    'pros': ['不增加征信查询记录', '审批更快'],
                    'cons': ['可能费率稍高']
                })
        
        if action_type == ActionType.HIGH_UTILIZATION:
            alternatives.append({
                'type': '分批还款',
                'description': '在账单日前分多次还款降低使用率',
                'pros': ['立即改善信用表现', '无额外成本'],
                'cons': ['需要提前规划']
            })
        
        return alternatives
    
    def _prepare_visualization_data(self, current_state: Dict, 
                                    impact_simulation: Dict) -> Dict:
        return {
            'score_trend': {
                'current': impact_simulation['current_score'],
                'projected': impact_simulation['projected_score'],
                'projections': impact_simulation['time_projections']
            },
            'risk_factors': {
                'utilization': current_state['credit_utilization'],
                'inquiries': current_state['hard_inquiries_6m'],
                'debt_ratio': current_state['total_debt'] / 50000
            }
        }
    
    async def _send_warning(self, user_id: str, warning_message: Dict):
        await self.notification_service.send_notification(
            user_id=user_id,
            notification_type='credit_warning',
            content=warning_message,
            priority=warning_message['risk_level']
        )
    
    async def _log_action(self, user_id: str, action_type: ActionType, 
                         risk_assessment: Dict, warning_sent: bool):
        await self.database.log_user_action(
            user_id=user_id,
            action_type=action_type.value,
            risk_score=risk_assessment['risk_score'],
            risk_level=risk_assessment['risk_level'].value,
            warning_sent=warning_sent,
            timestamp=datetime.now()
        )
    
    async def get_user_risk_summary(self, user_id: str) -> Dict:
        recent_actions = list(self.user_action_history[user_id])[-30:]
        
        high_risk_count = sum(
            1 for action in recent_actions
            if action.get('risk_level') in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
        
        credit_state = await self._get_user_credit_state(user_id)
        
        return {
            'user_id': user_id,
            'current_score': credit_state['current_score'] if credit_state else 0,
            'recent_high_risk_actions': high_risk_count,
            'total_recent_actions': len(recent_actions),
            'overall_risk_level': self._calculate_overall_risk(recent_actions),
            'last_warning_date': self._get_last_warning_date(user_id)
        }
    
    def _calculate_overall_risk(self, recent_actions: List[Dict]) -> str:
        if not recent_actions:
            return 'low'
        
        high_risk = sum(1 for a in recent_actions 
                       if a.get('risk_level') == RiskLevel.HIGH)
        critical_risk = sum(1 for a in recent_actions 
                           if a.get('risk_level') == RiskLevel.CRITICAL)
        
        if critical_risk >= 2:
            return 'critical'
        elif high_risk >= 3 or critical_risk >= 1:
            return 'high'
        elif high_risk >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _get_last_warning_date(self, user_id: str) -> Optional[str]:
        actions = list(self.user_action_history[user_id])
        for action in reversed(actions):
            if action.get('warning_sent'):
                return action['timestamp'].isoformat()
        return None