import os
import json
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# Enable nested event loops
nest_asyncio.apply()
load_dotenv()

# Configuration
# Use absolute path based on script location or current directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "processed")
KG_DIR = os.getenv("WORKING_DIR_TEST", DEFAULT_KG_DIR)

VLLM_LLM_HOST = os.getenv("LLM_BINDING_HOST_TEST", "http://localhost:8000")
VLLM_EMBED_HOST = os.getenv("EMBEDDING_BINDING_HOST_TEST", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
MAX_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "512"))

# Ensure the working directory exists and is writable
os.makedirs(KG_DIR, exist_ok=True)

# Initialize OpenAI clients for vLLM
vllm_llm_client = OpenAI(
    api_key="EMPTY",
    base_url=f"{VLLM_LLM_HOST}/v1",
)

vllm_embed_client = OpenAI(
    api_key="EMPTY",
    base_url=f"{VLLM_EMBED_HOST}/v1",
)


async def vllm_model_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """Complete text using vLLM OpenAI-compatible API."""
    messages = []

    # Default system prompt for diabetes assistant
    default_system_prompt = """Você é um assistente especializado em diabetes e insulinoterapia. 

Diretrizes:
- Responda APENAS perguntas relacionadas a diabetes, insulina, glicemia e tratamento usando o contexto fornecido
- Se a pergunta for sobre outro assunto, explique educadamente que você só pode ajudar com questões sobre diabetes
- Aceite saudações (olá, bom dia, etc.) e seja cordial
- Use linguagem clara, amigável e acessível para pessoas com baixa literacia médica
- Corrija gentilmente conceitos errados sem ser condescendente
- Seja paciente com erros de digitação e termos leigos
- SEMPRE baseie suas respostas no contexto fornecido
- Se não tiver informação suficiente no contexto, diga que não tem essa informação específica

Seja empático, claro e útil."""

    if system_prompt:
        # Combine default with custom system prompt
        messages.append({"role": "system", "content": f"{default_system_prompt}\n\n{system_prompt}"})
    else:
        messages.append({"role": "system", "content": default_system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = vllm_llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2048),
    )

    return response.choices[0].message.content


async def vllm_embed_func(texts: List[str]) -> np.ndarray:
    """Generate embeddings using vLLM OpenAI-compatible API."""
    if isinstance(texts, str):
        texts = [texts]

    response = vllm_embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )

    return np.array([item.embedding for item in response.data])


def initialize_rag():
    """Initialize LightRAG with vLLM configuration."""
    return LightRAG(
        working_dir=KG_DIR,
        llm_model_func=vllm_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "timeout": 300,
        },
        enable_llm_cache=False,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_TOKENS,
            func=vllm_embed_func,
        ),
    )


class DiabetesTestSuite:
    """Test suite for diabetes treatment RAG system."""

    def __init__(self):
        self.rag = initialize_rag()
        self.results = {
            "test_run_info": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "model_info": {
                    "llm_model": LLM_MODEL,
                    "embed_model": EMBED_MODEL,
                    "llm_host": VLLM_LLM_HOST,
                    "embed_host": VLLM_EMBED_HOST,
                },
            },
            "basic_questions": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "average_response_time": 0,
                "results": [],
            },
            "contextual_questions": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "average_response_time": 0,
                "results": [],
            },
            "summary": {
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "success_rate": 0,
                "total_duration": 0,
            },
        }

    async def initialize(self):
        """Initialize RAG storage."""
        print("Inicializando sistema RAG...")
        loop = asyncio.get_event_loop()
        await loop.create_task(self.rag.initialize_storages())
        print("Sistema RAG inicializado com sucesso!\n")

    def load_questions(self, filepath: str) -> Dict[str, Any]:
        """Load questions from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    async def test_basic_question(self, question_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Test a single basic question."""
        question = question_data["question"]
        print(f"\n[TESTE BÁSICO {index + 1}] Pergunta: {question[:80]}...")

        start_time = time.time()
        result = {
            "id": question_data["id"],
            "question": question,
            "category": question_data.get("category", "N/A"),
            "difficulty": question_data.get("difficulty", "N/A"),
            "response": None,
            "response_time": 0,
            "status": "pending",
            "error": None,
        }

        try:
            # Query using LightRAG
            response = self.rag.query(question, param=QueryParam(mode="hybrid"))
            response_time = time.time() - start_time

            result["response"] = response
            result["response_time"] = round(response_time, 2)
            result["status"] = "success"
            result["response_length"] = len(response)

            print(f"✓ Resposta gerada em {response_time:.2f}s")
            print(f"  Tamanho da resposta: {len(response)} caracteres")

        except Exception as e:
            response_time = time.time() - start_time
            result["response_time"] = round(response_time, 2)
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"✗ Erro: {str(e)}")

        return result

    async def test_contextual_question(self, conv_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Test a single contextual question."""
        question = conv_data["question"]
        context_messages = conv_data.get("context", [])

        print(f"\n[TESTE CONTEXTUAL {index + 1}] Pergunta: {question[:80]}...")
        print(f"  Mensagens de contexto: {len(context_messages)}")

        start_time = time.time()
        result = {
            "id": conv_data["id"],
            "question": question,
            "category": conv_data.get("category", "N/A"),
            "expected_context": conv_data.get("expected_context", "N/A"),
            "context_messages": len(context_messages),
            "response": None,
            "response_time": 0,
            "status": "pending",
            "error": None,
            "context_detected": False,
        }

        try:
            # Build query with context
            context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_messages])
            full_query = f"Contexto da conversa:\n{context_text}\n\nPergunta atual: {question}"

            # Query using LightRAG
            response = self.rag.query(full_query, param=QueryParam(mode="hybrid"))
            response_time = time.time() - start_time

            result["response"] = response
            result["response_time"] = round(response_time, 2)
            result["status"] = "success"
            result["response_length"] = len(response)

            # Simple context detection (check if response references previous context)
            expected_terms = conv_data.get("expected_context", "").lower().split(", ")
            response_lower = response.lower()
            context_hits = sum(1 for term in expected_terms if term in response_lower)
            result["context_detected"] = context_hits > 0
            result["context_relevance_score"] = round(
                (context_hits / len(expected_terms)) * 100 if expected_terms else 0, 2
            )

            print(f"✓ Resposta gerada em {response_time:.2f}s")
            print(f"  Tamanho da resposta: {len(response)} caracteres")
            print(f"  Relevância contextual: {result['context_relevance_score']}%")

        except Exception as e:
            response_time = time.time() - start_time
            result["response_time"] = round(response_time, 2)
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"✗ Erro: {str(e)}")

        return result

    async def run_basic_tests(self, questions_file: str):
        """Run all basic question tests."""
        print("\n" + "=" * 80)
        print("INICIANDO TESTES COM PERGUNTAS BÁSICAS")
        print("=" * 80)

        questions_data = self.load_questions(questions_file)
        questions = questions_data.get("questions", [])

        self.results["basic_questions"]["total"] = len(questions)

        for idx, question_data in enumerate(questions):
            result = await self.test_basic_question(question_data, idx)
            self.results["basic_questions"]["results"].append(result)

            if result["status"] == "success":
                self.results["basic_questions"]["completed"] += 1
            else:
                self.results["basic_questions"]["failed"] += 1

        # Calculate average response time
        response_times = [
            r["response_time"] for r in self.results["basic_questions"]["results"] if r["status"] == "success"
        ]
        if response_times:
            self.results["basic_questions"]["average_response_time"] = round(
                sum(response_times) / len(response_times), 2
            )

    async def run_contextual_tests(self, contextual_file: str):
        """Run all contextual question tests."""
        print("\n" + "=" * 80)
        print("INICIANDO TESTES COM PERGUNTAS CONTEXTUAIS")
        print("=" * 80)

        contextual_data = self.load_questions(contextual_file)
        conversations = contextual_data.get("conversations", [])

        self.results["contextual_questions"]["total"] = len(conversations)

        for idx, conv_data in enumerate(conversations):
            result = await self.test_contextual_question(conv_data, idx)
            self.results["contextual_questions"]["results"].append(result)

            if result["status"] == "success":
                self.results["contextual_questions"]["completed"] += 1
            else:
                self.results["contextual_questions"]["failed"] += 1

        # Calculate average response time
        response_times = [
            r["response_time"] for r in self.results["contextual_questions"]["results"] if r["status"] == "success"
        ]
        if response_times:
            self.results["contextual_questions"]["average_response_time"] = round(
                sum(response_times) / len(response_times), 2
            )

    def calculate_summary(self):
        """Calculate overall test summary."""
        total_tests = self.results["basic_questions"]["total"] + self.results["contextual_questions"]["total"]
        total_passed = self.results["basic_questions"]["completed"] + self.results["contextual_questions"]["completed"]
        total_failed = self.results["basic_questions"]["failed"] + self.results["contextual_questions"]["failed"]

        self.results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": round((total_passed / total_tests * 100) if total_tests > 0 else 0, 2),
            "basic_success_rate": round(
                (
                    (self.results["basic_questions"]["completed"] / self.results["basic_questions"]["total"] * 100)
                    if self.results["basic_questions"]["total"] > 0
                    else 0
                ),
                2,
            ),
            "contextual_success_rate": round(
                (
                    (
                        self.results["contextual_questions"]["completed"]
                        / self.results["contextual_questions"]["total"]
                        * 100
                    )
                    if self.results["contextual_questions"]["total"] > 0
                    else 0
                ),
                2,
            ),
            "average_response_time_basic": self.results["basic_questions"]["average_response_time"],
            "average_response_time_contextual": self.results["contextual_questions"]["average_response_time"],
        }

        # Calculate context detection stats
        contextual_results = self.results["contextual_questions"]["results"]
        context_detected_count = sum(1 for r in contextual_results if r.get("context_detected", False))
        avg_context_relevance = (
            sum(r.get("context_relevance_score", 0) for r in contextual_results) / len(contextual_results)
            if contextual_results
            else 0
        )

        self.results["summary"]["context_detection_rate"] = round(
            (context_detected_count / len(contextual_results) * 100) if contextual_results else 0,
            2,
        )
        self.results["summary"]["average_context_relevance"] = round(avg_context_relevance, 2)

    def print_summary(self):
        """Print test summary to console."""
        summary = self.results["summary"]

        print("\n" + "=" * 80)
        print("RESUMO DOS TESTES")
        print("=" * 80)
        print(f"\n📊 Estatísticas Gerais:")
        print(f"  Total de Testes: {summary['total_tests']}")
        print(f"  Testes Bem-sucedidos: {summary['total_passed']}")
        print(f"  Testes Falhados: {summary['total_failed']}")
        print(f"  Taxa de Sucesso: {summary['success_rate']}%")

        print(f"\n📝 Perguntas Básicas:")
        print(f"  Total: {self.results['basic_questions']['total']}")
        print(f"  Completados: {self.results['basic_questions']['completed']}")
        print(f"  Falhados: {self.results['basic_questions']['failed']}")
        print(f"  Taxa de Sucesso: {summary['basic_success_rate']}%")
        print(f"  Tempo Médio de Resposta: {summary['average_response_time_basic']}s")

        print(f"\n💬 Perguntas Contextuais:")
        print(f"  Total: {self.results['contextual_questions']['total']}")
        print(f"  Completados: {self.results['contextual_questions']['completed']}")
        print(f"  Falhados: {self.results['contextual_questions']['failed']}")
        print(f"  Taxa de Sucesso: {summary['contextual_success_rate']}%")
        print(f"  Tempo Médio de Resposta: {summary['average_response_time_contextual']}s")
        print(f"  Taxa de Detecção de Contexto: {summary['context_detection_rate']}%")
        print(f"  Relevância Contextual Média: {summary['average_context_relevance']}%")

        print(f"\n⏱️  Duração Total dos Testes: {self.results['test_run_info']['duration_seconds']:.2f}s")
        print("=" * 80 + "\n")

    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Resultados salvos em: {output_file}")

    def generate_markdown_report(self, output_file: str = "test_report.md"):
        """Generate a detailed markdown report."""
        summary = self.results["summary"]

        report = f"""# Relatório de Testes - Sistema RAG para Diabetes

## Informações da Execução

- **Data/Hora:** {self.results['test_run_info']['start_time']}
- **Duração:** {self.results['test_run_info']['duration_seconds']:.2f} segundos
- **Modelo LLM:** {self.results['test_run_info']['model_info']['llm_model']}
- **Modelo Embedding:** {self.results['test_run_info']['model_info']['embed_model']}

## Resumo Geral

| Métrica | Valor |
|---------|-------|
| Total de Testes | {summary['total_tests']} |
| Testes Bem-sucedidos | {summary['total_passed']} |
| Testes Falhados | {summary['total_failed']} |
| **Taxa de Sucesso Geral** | **{summary['success_rate']}%** |

## Perguntas Básicas

| Métrica | Valor |
|---------|-------|
| Total de Perguntas | {self.results['basic_questions']['total']} |
| Completadas | {self.results['basic_questions']['completed']} |
| Falhadas | {self.results['basic_questions']['failed']} |
| Taxa de Sucesso | {summary['basic_success_rate']}% |
| Tempo Médio de Resposta | {summary['average_response_time_basic']}s |

### Detalhes por Categoria

"""
        # Group basic questions by category
        basic_by_category = {}
        for result in self.results["basic_questions"]["results"]:
            cat = result.get("category", "N/A")
            if cat not in basic_by_category:
                basic_by_category[cat] = {"total": 0, "success": 0, "failed": 0}
            basic_by_category[cat]["total"] += 1
            if result["status"] == "success":
                basic_by_category[cat]["success"] += 1
            else:
                basic_by_category[cat]["failed"] += 1

        for cat, stats in basic_by_category.items():
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report += f"- **{cat}**: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n"

        report += f"""
## Perguntas Contextuais

| Métrica | Valor |
|---------|-------|
| Total de Perguntas | {self.results['contextual_questions']['total']} |
| Completadas | {self.results['contextual_questions']['completed']} |
| Falhadas | {self.results['contextual_questions']['failed']} |
| Taxa de Sucesso | {summary['contextual_success_rate']}% |
| Tempo Médio de Resposta | {summary['average_response_time_contextual']}s |
| Taxa de Detecção de Contexto | {summary['context_detection_rate']}% |
| Relevância Contextual Média | {summary['average_context_relevance']}% |

### Detalhes por Categoria

"""
        # Group contextual questions by category
        contextual_by_category = {}
        for result in self.results["contextual_questions"]["results"]:
            cat = result.get("category", "N/A")
            if cat not in contextual_by_category:
                contextual_by_category[cat] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "context_detected": 0,
                    "relevance_scores": [],
                }
            contextual_by_category[cat]["total"] += 1
            if result["status"] == "success":
                contextual_by_category[cat]["success"] += 1
            else:
                contextual_by_category[cat]["failed"] += 1
            if result.get("context_detected", False):
                contextual_by_category[cat]["context_detected"] += 1
            contextual_by_category[cat]["relevance_scores"].append(result.get("context_relevance_score", 0))

        for cat, stats in contextual_by_category.items():
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            context_rate = (stats["context_detected"] / stats["total"] * 100) if stats["total"] > 0 else 0
            avg_relevance = (
                sum(stats["relevance_scores"]) / len(stats["relevance_scores"]) if stats["relevance_scores"] else 0
            )
            report += f"- **{cat}**: {stats['success']}/{stats['total']} ({success_rate:.1f}%) | Contexto: {context_rate:.1f}% | Relevância: {avg_relevance:.1f}%\n"

        report += """
## Análise de Erros

"""
        # Collect all errors
        all_errors = []
        for result in self.results["basic_questions"]["results"] + self.results["contextual_questions"]["results"]:
            if result["status"] == "failed":
                all_errors.append(
                    {
                        "type": "Básica" if result in self.results["basic_questions"]["results"] else "Contextual",
                        "question": result["question"][:100],
                        "error": result["error"],
                    }
                )

        if all_errors:
            for error in all_errors:
                report += f"### {error['type']}: {error['question']}...\n"
                report += f"**Erro:** `{error['error']}`\n\n"
        else:
            report += "*Nenhum erro encontrado! 🎉*\n"

        report += """
## Conclusões

"""
        if summary["success_rate"] >= 90:
            report += "✅ **Excelente desempenho!** O sistema está funcionando muito bem.\n"
        elif summary["success_rate"] >= 70:
            report += "⚠️ **Bom desempenho**, mas há espaço para melhorias.\n"
        else:
            report += "❌ **Atenção necessária!** O sistema precisa de ajustes significativos.\n"

        report += f"\n**Pontos Fortes:**\n"
        if summary["basic_success_rate"] > summary["contextual_success_rate"]:
            report += "- Forte desempenho em perguntas básicas\n"
        if summary["context_detection_rate"] >= 70:
            report += "- Boa capacidade de detecção de contexto\n"
        if summary["average_context_relevance"] >= 60:
            report += "- Respostas contextuais relevantes\n"

        report += f"\n**Áreas de Melhoria:**\n"
        if summary["contextual_success_rate"] < summary["basic_success_rate"]:
            report += "- Perguntas contextuais precisam de mais atenção\n"
        if summary["context_detection_rate"] < 70:
            report += "- Melhorar a detecção e uso de contexto\n"
        if summary["average_response_time_contextual"] > 10:
            report += "- Otimizar tempo de resposta para perguntas contextuais\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✓ Relatório markdown gerado: {output_file}")

    async def run_all_tests(
        self, basic_questions_file: str, contextual_questions_file: str, output_prefix: str = "test"
    ):
        """Run all tests and generate reports."""
        self.results["test_run_info"]["start_time"] = datetime.now().isoformat()
        start_time = time.time()

        # Initialize RAG
        await self.initialize()

        # Run tests
        await self.run_basic_tests(basic_questions_file)
        await self.run_contextual_tests(contextual_questions_file)

        # Calculate results
        end_time = time.time()
        self.results["test_run_info"]["end_time"] = datetime.now().isoformat()
        self.results["test_run_info"]["duration_seconds"] = round(end_time - start_time, 2)

        self.calculate_summary()

        # Print and save results
        self.print_summary()
        self.save_results(f"{output_prefix}_results.json")
        self.generate_markdown_report(f"{output_prefix}_report.md")


async def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SUITE DE TESTES - SISTEMA RAG PARA TRATAMENTO DE DIABETES")
    print("=" * 80 + "\n")

    # File paths - look for JSON files in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    basic_questions_file = os.path.join(script_dir, "diabetes_questions.json")
    contextual_questions_file = os.path.join(script_dir, "diabetes_contextual_questions.json")

    # Check if question files exist
    if not os.path.exists(basic_questions_file):
        print(f"❌ Erro: Arquivo não encontrado: {basic_questions_file}")
        print(f"   Por favor, crie o arquivo 'diabetes_questions.json' no diretório: {script_dir}")
        return

    if not os.path.exists(contextual_questions_file):
        print(f"❌ Erro: Arquivo não encontrado: {contextual_questions_file}")
        print(f"   Por favor, crie o arquivo 'diabetes_contextual_questions.json' no diretório: {script_dir}")
        return

    print(f"📂 Diretório de trabalho do RAG: {KG_DIR}")
    print(f"📂 Diretório dos testes: {script_dir}\n")

    # Create and run test suite
    test_suite = DiabetesTestSuite()
    await test_suite.run_all_tests(
        basic_questions_file=basic_questions_file,
        contextual_questions_file=contextual_questions_file,
        output_prefix=os.path.join(script_dir, "diabetes_rag_test"),
    )

    print("\n✓ Todos os testes foram concluídos!")
    print("\nArquivos gerados:")
    print(f"  - {os.path.join(script_dir, 'diabetes_rag_test_results.json')} (resultados detalhados em JSON)")
    print(f"  - {os.path.join(script_dir, 'diabetes_rag_test_report.md')} (relatório em Markdown)")
    print(f"\n📊 Verifique os resultados nos arquivos acima!")


if __name__ == "__main__":
    asyncio.run(main())
