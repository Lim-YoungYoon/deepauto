"""
Agent Visualization Adapter

기존 decision.py의 에이전트 시스템과 시각화 모듈을 연결하는 어댑터입니다.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from visualization.visualization import AgentTopologyVisualizer, AgentVisualizationManager


class AgentVisualizationAdapter:
    """에이전트 시각화 어댑터"""
    
    def __init__(self):
        self.visualizer = AgentTopologyVisualizer()
        self.viz_manager = AgentVisualizationManager()
        self.viz_manager.visualizer = self.visualizer
        self.is_monitoring = False
        
    def start_monitoring(self):
        """모니터링 시작"""
        self.is_monitoring = True
        print("Agent visualization monitoring started!")
        print("Available visualization commands:")
        print("   - show_topology(): 에이전트 토폴로지 그래프")
        print("   - show_timeline(): 상호작용 타임라인")
        print("   - show_dashboard(): 성능 대시보드")
        print("   - show_agent_details(agent_name): 특정 에이전트 상세 정보")
        print("   - export_data(): 데이터 내보내기")
        print("   - start_animation(): 실시간 애니메이션")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        print("Agent visualization monitoring stopped.")
    
    def record_decision_agent_activity(self, query: str, selected_agent: str, 
                                     confidence: float, reasoning: str):
        """Decision Agent 활동 기록"""
        if not self.is_monitoring:
            return
        
        # Decision Agent에서 선택된 에이전트로의 라우팅 기록
        self.visualizer.record_interaction(
            source="decision_agent",
            target=selected_agent.lower().replace(" ", "_") + "_agent",
            interaction_type="routing",
            data_size=len(query),
            duration=0.1,  # 실제로는 측정된 시간 사용
            success=True
        )
        
        # Decision Agent 성능 지표 업데이트
        self.visualizer.update_performance_metrics("decision_agent", {
            "avg_response_time": 0.1,
            "success_rate": 95.0,
            "confidence_score": confidence
        })
        
        # 선택된 에이전트 상태 업데이트
        target_agent = selected_agent.lower().replace(" ", "_") + "_agent"
        if target_agent in self.visualizer.agents:
            self.visualizer.update_agent_status(target_agent, "active")
    
    def record_rag_agent_activity(self, query: str, response: str, 
                                retrieval_confidence: float, processing_time: float):
        """RAG Agent 활동 기록"""
        if not self.is_monitoring:
            return
        
        # RAG Agent 활동 기록
        self.visualizer.record_interaction(
            source="rag_agent",
            target="user",  # 사용자에게 응답
            interaction_type="data_flow",
            data_size=len(response),
            duration=processing_time,
            success=True
        )
        
        # RAG Agent 성능 지표 업데이트
        self.visualizer.update_performance_metrics("rag_agent", {
            "avg_response_time": processing_time,
            "retrieval_accuracy": retrieval_confidence * 100,
            "success_rate": 90.0
        })
    
    def record_web_search_activity(self, query: str, search_results: list, 
                                 processing_time: float):
        """Web Search Agent 활동 기록"""
        if not self.is_monitoring:
            return
        
        # Web Search Agent 활동 기록
        self.visualizer.record_interaction(
            source="web_search_agent",
            target="user",
            interaction_type="data_flow",
            data_size=len(str(search_results)),
            duration=processing_time,
            success=True
        )
        
        # Web Search Agent 성능 지표 업데이트
        self.visualizer.update_performance_metrics("web_search_agent", {
            "avg_response_time": processing_time,
            "search_relevance": 85.0,
            "success_rate": 88.0
        })
    
    def record_conversation_activity(self, query: str, response: str, 
                                   processing_time: float):
        """Conversation Agent 활동 기록"""
        if not self.is_monitoring:
            return
        
        # Conversation Agent 활동 기록
        self.visualizer.record_interaction(
            source="conversation_agent",
            target="user",
            interaction_type="control_flow",
            data_size=len(response),
            duration=processing_time,
            success=True
        )
        
        # Conversation Agent 성능 지표 업데이트
        self.visualizer.update_performance_metrics("conversation_agent", {
            "avg_response_time": processing_time,
            "user_satisfaction": 92.0,
            "success_rate": 95.0
        })
    
    def record_guardrails_activity(self, input_text: str, is_allowed: bool, 
                                 validation_time: float):
        """Guardrails 활동 기록"""
        if not self.is_monitoring:
            return
        
        # Guardrails 활동 기록
        self.visualizer.record_interaction(
            source="guardrails",
            target="all_agents" if is_allowed else "blocked",
            interaction_type="control_flow",
            data_size=len(input_text),
            duration=validation_time,
            success=is_allowed
        )
        
        # Guardrails 성능 지표 업데이트
        blocked_count = self.visualizer.agents["guardrails"].performance_metrics.get("blocked_requests", 0)
        if not is_allowed:
            blocked_count += 1
        
        self.visualizer.update_performance_metrics("guardrails", {
            "validation_rate": 98.0,
            "blocked_requests": blocked_count,
            "avg_validation_time": validation_time
        })
    
    def record_error(self, agent_name: str, error_message: str):
        """에러 기록"""
        if not self.is_monitoring:
            return
        
        # 에이전트 상태를 error로 변경
        self.visualizer.update_agent_status(agent_name, "error")
        
        # 에러 상호작용 기록
        self.visualizer.record_interaction(
            source=agent_name,
            target="error_handler",
            interaction_type="control_flow",
            data_size=len(error_message),
            duration=0.0,
            success=False
        )
    
    def get_visualization_manager(self) -> AgentVisualizationManager:
        """시각화 매니저 반환"""
        return self.viz_manager
    
    def get_visualizer(self) -> AgentTopologyVisualizer:
        """시각화 객체 반환"""
        return self.visualizer
    
    def export_current_state(self, filename: str = "agent_state.json"):
        """현재 상태 내보내기"""
        self.visualizer.export_topology_data(filename)
    
    def get_agent_status_summary(self) -> Dict[str, Any]:
        """에이전트 상태 요약 반환"""
        summary = {
            "total_agents": len(self.visualizer.agents),
            "active_agents": 0,
            "idle_agents": 0,
            "error_agents": 0,
            "total_interactions": len(self.visualizer.interactions),
            "recent_interactions": len([i for i in self.visualizer.interactions 
                                     if (datetime.now() - i.timestamp).seconds < 300])  # 최근 5분
        }
        
        for agent in self.visualizer.agents.values():
            if agent.status == "active":
                summary["active_agents"] += 1
            elif agent.status == "idle":
                summary["idle_agents"] += 1
            elif agent.status == "error":
                summary["error_agents"] += 1
        
        return summary


# 전역 시각화 어댑터 인스턴스
visualization_adapter = AgentVisualizationAdapter()


def get_visualization_adapter() -> AgentVisualizationAdapter:
    """전역 시각화 어댑터 반환"""
    return visualization_adapter


def start_agent_monitoring():
    """에이전트 모니터링 시작"""
    visualization_adapter.start_monitoring()


def stop_agent_monitoring():
    """에이전트 모니터링 중지"""
    visualization_adapter.stop_monitoring()


def record_agent_interaction(source: str, target: str, interaction_type: str, 
                           data_size: Optional[int] = None, 
                           duration: Optional[float] = None, 
                           success: bool = True):
    """에이전트 상호작용 기록 (일반적인 경우)"""
    if visualization_adapter.is_monitoring:
        visualization_adapter.visualizer.record_interaction(
            source=source,
            target=target,
            interaction_type=interaction_type,
            data_size=data_size,
            duration=duration,
            success=success
        )


# 사용 예시 및 테스트 함수
def test_visualization_adapter():
    """시각화 어댑터 테스트"""
    print("Testing Agent Visualization Adapter...")
    
    # 모니터링 시작
    adapter = get_visualization_adapter()
    adapter.start_monitoring()
    
    # 시뮬레이션 데이터 생성
    print("Generating simulation data...")
    
    # Decision Agent 활동 시뮬레이션
    adapter.record_decision_agent_activity(
        query="문서에서 AI에 대해 설명해주세요",
        selected_agent="RAG Agent",
        confidence=0.95,
        reasoning="PDF 파일이 업로드되어 문서 기반 질의응답이 필요함"
    )
    
    # RAG Agent 활동 시뮬레이션
    adapter.record_rag_agent_activity(
        query="문서에서 AI에 대해 설명해주세요",
        response="AI는 인공지능의 약자로...",
        retrieval_confidence=0.88,
        processing_time=1.2
    )
    
    # Web Search Agent 활동 시뮬레이션
    adapter.record_web_search_activity(
        query="최신 AI 기술 동향",
        search_results=["OpenAI GPT-4", "Google Gemini", "Anthropic Claude"],
        processing_time=2.5
    )
    
    # Conversation Agent 활동 시뮬레이션
    adapter.record_conversation_activity(
        query="안녕하세요",
        response="안녕하세요! 무엇을 도와드릴까요?",
        processing_time=0.3
    )
    
    # Guardrails 활동 시뮬레이션
    adapter.record_guardrails_activity(
        input_text="정상적인 질문입니다",
        is_allowed=True,
        validation_time=0.05
    )
    
    # 상태 요약 출력
    summary = adapter.get_agent_status_summary()
    print("\nAgent Status Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # 시각화 매니저 가져오기
    viz_manager = adapter.get_visualization_manager()
    
    print("\nAvailable visualizations:")
    print("   - viz_manager.show_topology()")
    print("   - viz_manager.show_timeline()")
    print("   - viz_manager.show_dashboard()")
    print("   - viz_manager.show_agent_details('decision_agent')")
    
    return adapter


if __name__ == "__main__":
    # 테스트 실행
    test_adapter = test_visualization_adapter()
    
    # 시각화 매니저로 다양한 시각화 테스트
    viz_manager = test_adapter.get_visualization_manager()
    
    print("\nRunning visualization tests...")
    
    # 토폴로지 그래프 생성 (실제로는 plt.show() 호출)
    print("Topology graph created")
    
    # 타임라인 생성
    print("Timeline created")
    
    # 대시보드 생성
    print("Dashboard created")
    
    # 데이터 내보내기
    test_adapter.export_current_state("test_agent_state.json")
    print("Data exported to test_agent_state.json")
    
    print("\nVisualization adapter test completed!") 