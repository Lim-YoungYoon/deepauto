/**
 * DeepAuto Chat Application JavaScript
 * Handles chat functionality, file uploads, and UI interactions
 */

class ChatApp {
    constructor() {
        this.displayMessages = [];
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // DOM elements
        this.chatBody = document.getElementById('chat-body');
        this.chatForm = document.getElementById('chat-form');
        this.messageInput = document.getElementById('message-input');
        this.pdfUpload = document.getElementById('pdf-upload');
        this.previewContainer = document.getElementById('preview-container');
        this.previewPdfName = document.getElementById('preview-pdf-name');
        this.removePdfBtn = document.getElementById('remove-pdf');
        this.clearChatBtn = document.getElementById('clear-chat');
        this.sidebar = document.getElementById('sidebar');
        this.sidebarOverlay = document.getElementById('sidebar-overlay');
        this.openSidebarBtn = document.getElementById('open-sidebar');
        this.closeSidebarBtn = document.getElementById('close-sidebar');
    }

    bindEvents() {
        // Sidebar events
        this.openSidebarBtn.addEventListener('click', () => this.toggleSidebar(true));
        this.closeSidebarBtn.addEventListener('click', () => this.toggleSidebar(false));
        this.sidebarOverlay.addEventListener('click', () => this.toggleSidebar(false));

        // Keyboard events
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.toggleSidebar(false);
            }
        });

        // Form events
        this.chatForm.addEventListener('submit', (e) => this.handleFormSubmit(e));
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());

        // File upload events
        this.pdfUpload.addEventListener('change', (e) => this.handleFileUpload(e));
        this.removePdfBtn.addEventListener('click', () => this.removeFile());

        // Clear chat event
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
    }

    toggleSidebar(show) {
        if (show) {
            this.sidebar.classList.add('active');
            this.sidebarOverlay.classList.add('active');
        } else {
            this.sidebar.classList.remove('active');
            this.sidebarOverlay.classList.remove('active');
        }
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = (this.messageInput.scrollHeight) + 'px';
    }

    handleFileUpload(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            this.previewPdfName.textContent = file.name;
            this.previewContainer.style.display = 'block';
            this.previewContainer.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <span id="preview-pdf-name">${file.name}</span>
                    <button type="button" class="btn btn-sm btn-outline-danger" id="remove-pdf">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            // Re-bind remove button event
            document.getElementById('remove-pdf').addEventListener('click', () => this.removeFile());
        }
    }

    removeFile() {
        this.pdfUpload.value = '';
        this.previewContainer.style.display = 'none';
        this.previewContainer.innerHTML = '';
    }

    async clearChat() {
        console.log('Clear chat button clicked');
        
        if (!confirm('정말로 모든 대화 내용과 업로드된 문서를 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.')) {
            console.log('User cancelled deletion');
            return;
        }
        
        console.log('User confirmed deletion, starting process...');
        
        // Show loading state
        const originalText = this.clearChatBtn.innerHTML;
        this.clearChatBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>삭제 중...';
        this.clearChatBtn.disabled = true;
        
        try {
            console.log('Sending request to /clear_data...');
            
            const response = await fetch('/clear_data', {
                method: 'POST',
                credentials: 'include'
            });
            
            console.log('Response received:', response.status);
            
            const result = await response.json();
            console.log('Response data:', result);
            
            if (result.success) {
                this.displayMessages = [];
                this.chatBody.innerHTML = `
                    <div class="message assistant-message">
                        <div class="agent-tag">System</div>
                        <p>모든 대화 내용과 문서가 삭제되었습니다. 새로운 대화를 시작하세요.</p>
                    </div>
                `;
                
                // Show success message with details
                const successMsg = document.createElement('div');
                successMsg.className = 'message assistant-message';
                let successHtml = `
                    <div class="agent-tag">System</div>
                    <p class="text-success">✅ ${result.message}</p>
                `;
                
                if (result.deleted_items && result.deleted_items.length > 0) {
                    successHtml += `
                        <div class="mt-2">
                            <small class="text-muted">삭제된 항목:</small>
                            <ul class="list-unstyled mt-1">
                                ${result.deleted_items.map(item => `<li><small>• ${item}</small></li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                successMsg.innerHTML = successHtml;
                this.chatBody.appendChild(successMsg);
                
                // Clear file upload preview
                this.removeFile();
                
            } else {
                this.showErrorMessage(`데이터 삭제 중 오류가 발생했습니다: ${result.error}`);
            }
            
        } catch (error) {
            console.error('데이터 삭제 오류:', error);
            console.error('Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
            
            this.showErrorMessage(`서버 연결 오류가 발생했습니다.\n오류: ${error.message}`);
        } finally {
            // Restore button state
            this.clearChatBtn.innerHTML = originalText;
            this.clearChatBtn.disabled = false;
            
            // Close sidebar after clearing chat
            this.toggleSidebar(false);
            
            // Scroll to bottom
            this.scrollToBottom();
        }
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.chatForm.dispatchEvent(new Event('submit'));
        }
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message && !this.pdfUpload.files.length) return;
        
        let pdfData = null;
        if (this.pdfUpload.files.length > 0) {
            pdfData = this.pdfUpload.files[0];
        }
        
        // Add user message to chat
        this.addUserMessage(message, pdfData);
        
        // Add thinking indicator
        const thinkingElement = this.addThinkingIndicator();
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        try {
            let response;
            
            if (this.pdfUpload.files.length > 0) {
                // Create form data for PDF upload with optional text
                const formData = new FormData();
                formData.append('text', message);
                formData.append('pdf', this.pdfUpload.files[0]);
                
                response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                    credentials: 'include'
                });
            } else {
                // Text-only query
                response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: message,
                        conversation_history: []
                    }),
                    credentials: 'include'
                });
            }
            
            const data = await response.json();
            
            // Remove thinking indicator
            this.chatBody.removeChild(thinkingElement);
            
            // Add assistant response
            this.addAssistantMessage(data, pdfData);
            
            // Store messages for display purposes
            this.displayMessages.push({
                role: "user",
                content: message,
                pdf: pdfData
            });
            
            this.displayMessages.push({
                role: "assistant",
                content: data.response,
                agent: data.agent,
                resultImage: data.result_image
            });
            
            // Clear file upload preview
            this.removeFile();
            
        } catch (error) {
            console.error("Error:", error);
            this.chatBody.removeChild(thinkingElement);
            this.showErrorMessage("요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.");
        }
        
        this.scrollToBottom();
    }

    addUserMessage(message, pdfData) {
        const userMessageElement = document.createElement('div');
        userMessageElement.className = 'message user-message';
        
        let messageContent = `<p>${message}</p>`;
        if (pdfData) {
            messageContent += `
                <div class="mt-2">
                    <p>Uploaded PDF: ${pdfData.name}</p>
                </div>
            `;
        }
        
        userMessageElement.innerHTML = messageContent;
        this.chatBody.appendChild(userMessageElement);
    }

    addThinkingIndicator() {
        const thinkingElement = document.createElement('div');
        thinkingElement.className = 'message assistant-message thinking';
        thinkingElement.innerHTML = `
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        `;
        this.chatBody.appendChild(thinkingElement);
        return thinkingElement;
    }

    addAssistantMessage(data, pdfData) {
        const assistantMessageElement = document.createElement('div');
        assistantMessageElement.className = 'message assistant-message';
        
        let assistantHtml = `
            <div class="agent-tag">${data.agent}</div>
            <div>${marked.parse(data.response)}</div>
        `;
        
        // Add result image if it exists
        if (data.result_image) {
            if (data.agent === "DOCUMENT_ANALYSIS_AGENT, HUMAN_VALIDATION" && pdfData) {
                assistantHtml += `
                    <div class="image-side-by-side">
                        <div class="image-container">
                            <p>Uploaded PDF: ${pdfData.name}</p>
                        </div>
                        <div class="image-container">
                            <img src="${data.result_image}" alt="Analysis result" class="img-fluid rounded">
                            <div class="image-caption">Analysis Result</div>
                        </div>
                    </div>
                `;
            } else {
                assistantHtml += `
                    <div class="mt-3">
                        <img src="${data.result_image}" alt="Result image" class="result-image">
                    </div>
                `;
            }
        }
        
        assistantMessageElement.innerHTML = assistantHtml;
        this.chatBody.appendChild(assistantMessageElement);

    }

    showErrorMessage(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'message assistant-message';
        errorElement.innerHTML = `
            <div class="agent-tag">System</div>
            <p class="text-danger">${message}</p>
        `;
        this.chatBody.appendChild(errorElement);
    }

    scrollToBottom() {
        this.chatBody.scrollTop = this.chatBody.scrollHeight;
    }
}

// Initialize the chat application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new ChatApp();
}); 