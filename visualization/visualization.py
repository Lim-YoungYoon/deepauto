"""
Agent Topology and Interaction Visualization Module

이 모듈은 멀티 에이전트 시스템의 토폴로지와 상호작용을 시각화합니다.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import defaultdict, deque
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class AgentNode:
    """에이전트 노드 정보"""
    name: str
    agent_type: str
    description: str
    capabilities: List[str]
    dependencies: List[str]
    position: Tuple[float, float] = (0, 0)
    status: str = "idle"  # idle, active, error
    last_activity: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None


@dataclass
class Interaction:
    """에이전트 간 상호작용 정보"""
    source: str
    target: str
    interaction_type: str  # routing, data_flow, control_flow
    timestamp: datetime
    data_size: Optional[int] = None
    duration: Optional[float] = None
    success: bool = True


class AgentTopologyVisualizer:
    """에이전트 토폴로지 시각화 클래스"""
    
    def __init__(self):
        self.agents = {}
        self.interactions = []
        self.interaction_history = deque(maxlen=1000)
        self.performance_history = defaultdict(list)
        self.setup_agents()
        
    def setup_agents(self):
        """에이전트 정보 초기화"""
        self.agents = {
            "decision_agent": AgentNode(
                name="Decision Agent",
                agent_type="router",
                description="사용자 쿼리를 적절한 에이전트로 라우팅",
                capabilities=["query_analysis", "agent_selection", "confidence_scoring"],
                dependencies=[],
                position=(0, 2),
                performance_metrics={"avg_response_time": 0.0, "success_rate": 0.0}
            ),
            "rag_agent": AgentNode(
                name="RAG Agent",
                agent_type="document_qa",
                description="문서 기반 질의응답 처리",
                capabilities=["document_parsing", "vector_search", "context_generation"],
                dependencies=["decision_agent"],
                position=(-2, 0),
                performance_metrics={"avg_response_time": 0.0, "retrieval_accuracy": 0.0}
            ),
            "web_search_agent": AgentNode(
                name="Web Search Agent",
                agent_type="web_search",
                description="웹 검색 기반 최신 정보 제공",
                capabilities=["web_search", "content_filtering", "information_synthesis"],
                dependencies=["decision_agent"],
                position=(2, 0),
                performance_metrics={"avg_response_time": 0.0, "search_relevance": 0.0}
            ),
            "conversation_agent": AgentNode(
                name="Conversation Agent",
                agent_type="chat",
                description="일반 대화 및 상호작용",
                capabilities=["natural_language", "context_awareness", "conversation_flow"],
                dependencies=["decision_agent"],
                position=(0, -2),
                performance_metrics={"avg_response_time": 0.0, "user_satisfaction": 0.0}
            ),
            "guardrails": AgentNode(
                name="Guardrails",
                agent_type="safety",
                description="입력/출력 검증 및 안전성 보장",
                capabilities=["input_validation", "output_sanitization", "content_filtering"],
                dependencies=[],
                position=(0, 4),
                performance_metrics={"validation_rate": 0.0, "blocked_requests": 0.0}
            )
        }
    
    def record_interaction(self, source: str, target: str, interaction_type: str, 
                          data_size: Optional[int] = None, duration: Optional[float] = None, 
                          success: bool = True):
        """에이전트 간 상호작용 기록"""
        interaction = Interaction(
            source=source,
            target=target,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            data_size=data_size,
            duration=duration,
            success=success
        )
        self.interactions.append(interaction)
        self.interaction_history.append(interaction)
        
        # 에이전트 상태 업데이트
        if source in self.agents:
            self.agents[source].last_activity = datetime.now()
            self.agents[source].status = "active"
        
        if target in self.agents:
            self.agents[target].last_activity = datetime.now()
            self.agents[target].status = "active"
    
    def update_agent_status(self, agent_name: str, status: str):
        """에이전트 상태 업데이트"""
        if agent_name in self.agents:
            self.agents[agent_name].status = status
            self.agents[agent_name].last_activity = datetime.now()
    
    def update_performance_metrics(self, agent_name: str, metrics: Dict[str, float]):
        """에이전트 성능 지표 업데이트"""
        if agent_name in self.agents:
            self.agents[agent_name].performance_metrics.update(metrics)
            for key, value in metrics.items():
                self.performance_history[f"{agent_name}_{key}"].append(value)
    
    def create_topology_graph(self, figsize=(12, 10)) -> plt.Figure:
        """에이전트 토폴로지 그래프 생성"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        # 노드 추가
        for agent_id, agent in self.agents.items():
            G.add_node(agent_id, **asdict(agent))
        
        # 엣지 추가 (의존성 기반)
        edges = []
        for agent_id, agent in self.agents.items():
            for dep in agent.dependencies:
                if dep in self.agents:
                    edges.append((dep, agent_id))
        
        G.add_edges_from(edges)
        
        # 노드 위치 설정
        pos = {agent_id: agent.position for agent_id, agent in self.agents.items()}
        
        # 노드 색상 매핑
        status_colors = {
            "idle": "#E8E8E8",
            "active": "#4CAF50",
            "error": "#F44336"
        }
        
        node_colors = [status_colors.get(self.agents[node].status, "#E8E8E8") 
                      for node in G.nodes()]
        
        # 그래프 그리기
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, 
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', ax=ax)
        
        # 노드 라벨
        labels = {node: self.agents[node].name for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
        
        # 엣지 라벨 (상호작용 타입)
        edge_labels = {}
        for edge in G.edges():
            recent_interactions = [i for i in self.interactions[-10:] 
                                 if i.source == edge[0] and i.target == edge[1]]
            if recent_interactions:
                edge_labels[edge] = f"{len(recent_interactions)} interactions"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Multi-Agent System Topology", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 범례 추가
        legend_elements = [patches.Patch(color=color, label=status) 
                          for status, color in status_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', title="Agent Status")
        
        return fig
    
    def create_interaction_timeline(self, figsize=(14, 8)) -> plt.Figure:
        """상호작용 타임라인 시각화"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if not self.interactions:
            ax.text(0.5, 0.5, "No interactions recorded yet", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Interaction Timeline")
            return fig
        
        # 최근 상호작용 필터링 (최근 50개)
        recent_interactions = self.interactions[-50:]
        
        # 타임라인 데이터 준비
        timestamps = [i.timestamp for i in recent_interactions]
        sources = [i.source for i in recent_interactions]
        targets = [i.target for i in recent_interactions]
        types = [i.interaction_type for i in recent_interactions]
        
        # 색상 매핑
        type_colors = {
            "routing": "#FF6B6B",
            "data_flow": "#4ECDC4",
            "control_flow": "#45B7D1"
        }
        
        colors = [type_colors.get(t, "#95A5A6") for t in types]
        
        # 산점도로 상호작용 시각화
        y_positions = list(range(len(recent_interactions)))
        ax.scatter(timestamps, y_positions, c=colors, s=100, alpha=0.7)
        
        # 상호작용 라벨
        for i, (timestamp, source, target) in enumerate(zip(timestamps, sources, targets)):
            ax.annotate(f"{source} → {target}", 
                       (timestamp, i), 
                       xytext=(5, 0), 
                       textcoords='offset points',
                       fontsize=8)
        
        ax.set_title("Recent Agent Interactions", fontsize=14, fontweight='bold')
        ax.set_ylabel("Interaction Index")
        ax.set_xlabel("Time")
        
        # 범례
        legend_elements = [patches.Patch(color=color, label=interaction_type) 
                          for interaction_type, color in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # x축 포맷팅
        ax.tick_params(axis='x', rotation=45)
        
        return fig
    
    def create_performance_dashboard(self, figsize=(16, 10)) -> plt.Figure:
        """성능 대시보드 생성"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Agent Performance Dashboard", fontsize=16, fontweight='bold')
        
        # 1. 응답 시간 비교
        ax1 = axes[0, 0]
        response_times = []
        agent_names = []
        
        for agent_id, agent in self.agents.items():
            if "avg_response_time" in agent.performance_metrics:
                response_times.append(agent.performance_metrics["avg_response_time"])
                agent_names.append(agent.name)
        
        if response_times:
            bars = ax1.bar(agent_names, response_times, color='skyblue', alpha=0.7)
            ax1.set_title("Average Response Time")
            ax1.set_ylabel("Time (seconds)")
            ax1.tick_params(axis='x', rotation=45)
            
            # 값 라벨 추가
            for bar, value in zip(bars, response_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}s', ha='center', va='bottom')
        
        # 2. 성공률
        ax2 = axes[0, 1]
        success_rates = []
        success_agent_names = []
        
        for agent_id, agent in self.agents.items():
            if "success_rate" in agent.performance_metrics:
                success_rates.append(agent.performance_metrics["success_rate"])
                success_agent_names.append(agent.name)
        
        if success_rates:
            bars = ax2.bar(success_agent_names, success_rates, color='lightgreen', alpha=0.7)
            ax2.set_title("Success Rate")
            ax2.set_ylabel("Rate (%)")
            ax2.tick_params(axis='x', rotation=45)
            
            # 값 라벨 추가
            for bar, value in zip(bars, success_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. 상호작용 빈도 히트맵
        ax3 = axes[1, 0]
        if len(self.interactions) > 0:
            # 상호작용 매트릭스 생성
            agent_ids = list(self.agents.keys())
            interaction_matrix = np.zeros((len(agent_ids), len(agent_ids)))
            
            for interaction in self.interactions:
                if interaction.source in agent_ids and interaction.target in agent_ids:
                    i = agent_ids.index(interaction.source)
                    j = agent_ids.index(interaction.target)
                    interaction_matrix[i, j] += 1
            
            im = ax3.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
            ax3.set_title("Interaction Frequency Heatmap")
            ax3.set_xticks(range(len(agent_ids)))
            ax3.set_yticks(range(len(agent_ids)))
            ax3.set_xticklabels([self.agents[aid].name for aid in agent_ids], rotation=45)
            ax3.set_yticklabels([self.agents[aid].name for aid in agent_ids])
            
            # 색상바 추가
            plt.colorbar(im, ax=ax3)
        
        # 4. 최근 활동 상태
        ax4 = axes[1, 1]
        status_counts = defaultdict(int)
        for agent in self.agents.values():
            status_counts[agent.status] += 1
        
        if status_counts:
            colors = ['#4CAF50', '#FF9800', '#F44336']
            wedges, texts, autotexts = ax4.pie(status_counts.values(), 
                                              labels=status_counts.keys(),
                                              autopct='%1.1f%%',
                                              colors=colors[:len(status_counts)])
            ax4.set_title("Agent Status Distribution")
        
        plt.tight_layout()
        return fig
    
    def create_agent_details_view(self, agent_name: str, figsize=(12, 8)) -> plt.Figure:
        """특정 에이전트 상세 정보 뷰"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Agent Details: {agent.name}", fontsize=16, fontweight='bold')
        
        # 1. 에이전트 정보
        ax1 = axes[0, 0]
        ax1.axis('off')
        info_text = f"""
        Type: {agent.agent_type}
        Status: {agent.status}
        Last Activity: {agent.last_activity or 'Never'}
        
        Capabilities:
        {chr(10).join(f'• {cap}' for cap in agent.capabilities)}
        
        Dependencies:
        {chr(10).join(f'• {dep}' for dep in agent.dependencies) if agent.dependencies else 'None'}
        """
        ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # 2. 성능 지표
        ax2 = axes[0, 1]
        if agent.performance_metrics:
            metrics = list(agent.performance_metrics.keys())
            values = list(agent.performance_metrics.values())
            
            bars = ax2.bar(metrics, values, color='lightblue', alpha=0.7)
            ax2.set_title("Performance Metrics")
            ax2.tick_params(axis='x', rotation=45)
            
            # 값 라벨 추가
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 3. 최근 상호작용
        ax3 = axes[1, 0]
        recent_interactions = [i for i in self.interactions[-20:] 
                             if i.source == agent_name or i.target == agent_name]
        
        if recent_interactions:
            interaction_types = [i.interaction_type for i in recent_interactions]
            type_counts = defaultdict(int)
            for itype in interaction_types:
                type_counts[itype] += 1
            
            if type_counts:
                ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                ax3.set_title("Recent Interaction Types")
        else:
            ax3.text(0.5, 0.5, "No recent interactions", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Recent Interaction Types")
        
        # 4. 시간별 활동
        ax4 = axes[1, 1]
        if agent.last_activity:
            # 최근 24시간 동안의 활동 시간 분포 (시뮬레이션)
            hours = list(range(24))
            activity_levels = np.random.randint(0, 10, 24)  # 실제로는 실제 데이터 사용
            
            ax4.bar(hours, activity_levels, color='orange', alpha=0.7)
            ax4.set_title("Activity by Hour (Last 24h)")
            ax4.set_xlabel("Hour")
            ax4.set_ylabel("Activity Level")
            ax4.set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        return fig
    
    def export_topology_data(self, filename: str = "agent_topology.json"):
        """토폴로지 데이터를 JSON으로 내보내기"""
        export_data = {
            "agents": {k: asdict(v) for k, v in self.agents.items()},
            "interactions": [asdict(i) for i in self.interactions[-100:]],  # 최근 100개만
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Topology data exported to {filename}")
    
    def create_animated_topology(self, duration: int = 10, interval: int = 1000) -> FuncAnimation:
        """애니메이션 토폴로지 생성"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame):
            ax.clear()
            
            # 랜덤 상호작용 시뮬레이션
            if frame % 5 == 0:  # 5프레임마다 새로운 상호작용
                agents = list(self.agents.keys())
                if len(agents) >= 2:
                    source = np.random.choice(agents)
                    target = np.random.choice([a for a in agents if a != source])
                    self.record_interaction(source, target, "simulated")
            
            # 토폴로지 그리기
            G = nx.DiGraph()
            for agent_id, agent in self.agents.items():
                G.add_node(agent_id, **asdict(agent))
            
            # 엣지 추가
            edges = []
            for agent_id, agent in self.agents.items():
                for dep in agent.dependencies:
                    if dep in self.agents:
                        edges.append((dep, agent_id))
            
            G.add_edges_from(edges)
            pos = {agent_id: agent.position for agent_id, agent in self.agents.items()}
            
            # 노드 색상 (활동 상태에 따라)
            status_colors = {"idle": "#E8E8E8", "active": "#4CAF50", "error": "#F44336"}
            node_colors = [status_colors.get(self.agents[node].status, "#E8E8E8") 
                          for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, 
                                  alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                  arrowsize=20, arrowstyle='->', ax=ax)
            
            labels = {node: self.agents[node].name for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title(f"Live Agent Topology - Frame {frame}", fontsize=14)
            ax.axis('off')
        
        anim = FuncAnimation(fig, animate, frames=duration, interval=interval, repeat=True)
        return anim


class AgentVisualizationManager:
    """에이전트 시각화 관리자"""
    
    def __init__(self):
        self.visualizer = AgentTopologyVisualizer()
        self.active_animations = []
    
    def start_monitoring(self):
        """모니터링 시작"""
        print("Agent visualization monitoring started...")
        print("Available commands:")
        print("- show_topology(): 토폴로지 그래프 표시")
        print("- show_timeline(): 상호작용 타임라인 표시")
        print("- show_dashboard(): 성능 대시보드 표시")
        print("- show_agent_details(agent_name): 특정 에이전트 상세 정보")
        print("- export_data(): 데이터 내보내기")
        print("- start_animation(): 애니메이션 시작")
    
    def show_topology(self):
        """토폴로지 그래프 표시"""
        fig = self.visualizer.create_topology_graph()
        plt.show()
    
    def show_timeline(self):
        """상호작용 타임라인 표시"""
        fig = self.visualizer.create_interaction_timeline()
        plt.show()
    
    def show_dashboard(self):
        """성능 대시보드 표시"""
        fig = self.visualizer.create_performance_dashboard()
        plt.show()
    
    def show_agent_details(self, agent_name: str):
        """특정 에이전트 상세 정보 표시"""
        try:
            fig = self.visualizer.create_agent_details_view(agent_name)
            plt.show()
        except ValueError as e:
            print(f"Error: {e}")
    
    def export_data(self, filename: str = "agent_topology.json"):
        """데이터 내보내기"""
        self.visualizer.export_topology_data(filename)
    
    def start_animation(self, duration: int = 10):
        """애니메이션 시작"""
        anim = self.visualizer.create_animated_topology(duration)
        plt.show()
        self.active_animations.append(anim)
    
    def simulate_interactions(self, num_interactions: int = 10):
        """상호작용 시뮬레이션"""
        agents = list(self.visualizer.agents.keys())
        interaction_types = ["routing", "data_flow", "control_flow"]
        
        for _ in range(num_interactions):
            source = np.random.choice(agents)
            target = np.random.choice([a for a in agents if a != source])
            interaction_type = np.random.choice(interaction_types)
            
            self.visualizer.record_interaction(
                source=source,
                target=target,
                interaction_type=interaction_type,
                data_size=np.random.randint(100, 10000),
                duration=np.random.uniform(0.1, 2.0),
                success=np.random.choice([True, False], p=[0.9, 0.1])
            )
            
            # 에이전트 성능 지표 업데이트
            for agent_id in [source, target]:
                if agent_id in self.visualizer.agents:
                    self.visualizer.update_performance_metrics(agent_id, {
                        "avg_response_time": np.random.uniform(0.5, 3.0),
                        "success_rate": np.random.uniform(85, 99)
                    })
        
        print(f"Simulated {num_interactions} interactions")


# 사용 예시
if __name__ == "__main__":
    # 시각화 매니저 초기화
    viz_manager = AgentVisualizationManager()
    
    # 모니터링 시작
    viz_manager.start_monitoring()
    
    # 시뮬레이션 데이터 생성
    viz_manager.simulate_interactions(20)
    
    # 다양한 시각화 표시
    print("\n=== 토폴로지 그래프 ===")
    viz_manager.show_topology()
    
    print("\n=== 상호작용 타임라인 ===")
    viz_manager.show_timeline()
    
    print("\n=== 성능 대시보드 ===")
    viz_manager.show_dashboard()
    
    print("\n=== Decision Agent 상세 정보 ===")
    viz_manager.show_agent_details("decision_agent")
    
    # 데이터 내보내기
    viz_manager.export_data() 