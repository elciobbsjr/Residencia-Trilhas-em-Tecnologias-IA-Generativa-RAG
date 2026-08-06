import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# Carrega as variáveis definidas no arquivo .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENAI_MODEL", "openrouter/free")

# Pasta que contém os arquivos Markdown
PASTA_MARKDOWN = Path("aula_2")

# Endpoint da API do OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Esquema que a inteligência artificial deve obrigatoriamente seguir
ESQUEMA_METADADOS = {
    "name": "metadados_documento",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "titulo": {
                "type": "string",
                "description": "Título completo do trabalho"
            },
            "autores": {
                "type": "array",
                "description": "Lista com os nomes dos autores",
                "items": {
                    "type": "string"
                }
            },
            "ano": {
                "type": ["integer", "null"],
                "description": "Ano de publicação do trabalho"
            }
        },
        "required": [
            "titulo",
            "autores",
            "ano"
        ],
        "additionalProperties": False
    }
}


def limpar_bloco_json(conteudo: str) -> str:
    """
    Remove blocos Markdown como ```json ... ```,
    caso algum modelo acrescente essa formatação.
    """
    conteudo = conteudo.strip()

    conteudo = re.sub(
        r"^```(?:json)?\s*",
        "",
        conteudo,
        flags=re.IGNORECASE
    )

    conteudo = re.sub(
        r"\s*```$",
        "",
        conteudo
    )

    return conteudo.strip()


def validar_metadados(dados: dict[str, Any]) -> dict[str, Any]:
    """
    Realiza uma validação adicional antes de salvar o JSON.
    """
    campos_obrigatorios = {"titulo", "autores", "ano"}

    campos_ausentes = campos_obrigatorios - dados.keys()

    if campos_ausentes:
        raise ValueError(
            f"Campos ausentes no resultado: {', '.join(campos_ausentes)}"
        )

    if not isinstance(dados["titulo"], str):
        raise TypeError("O campo 'titulo' deve ser uma string.")

    if not isinstance(dados["autores"], list):
        raise TypeError("O campo 'autores' deve ser uma lista.")

    if not all(isinstance(autor, str) for autor in dados["autores"]):
        raise TypeError("Todos os autores devem ser strings.")

    if dados["ano"] is not None and not isinstance(dados["ano"], int):
        raise TypeError("O campo 'ano' deve ser um número inteiro ou null.")

    return {
        "titulo": dados["titulo"].strip(),
        "autores": [
            autor.strip()
            for autor in dados["autores"]
            if autor.strip()
        ],
        "ano": dados["ano"]
    }


def extrair_metadados(conteudo_markdown: str) -> dict[str, Any]:
    """
    Recebe o conteúdo de um arquivo Markdown e retorna
    os metadados do trabalho em um dicionário Python.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "A variável OPENROUTER_API_KEY não foi encontrada no arquivo .env."
        )

    if not conteudo_markdown.strip():
        raise ValueError("O arquivo Markdown está vazio.")

    cabecalhos = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    corpo_requisicao = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você extrai metadados de trabalhos acadêmicos. "
                    "Use exclusivamente as informações presentes no documento. "
                    "Não invente título, autores ou ano. "
                    "Caso o ano não seja encontrado, retorne null. "
                    "Caso os autores não sejam encontrados, retorne uma lista vazia."
                )
            },
            {
                "role": "user",
                "content": (
                    "Extraia o título, os autores e o ano de publicação "
                    "do documento Markdown abaixo:\n\n"
                    f"{conteudo_markdown}"
                )
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": ESQUEMA_METADADOS
        },
        "temperature": 0
    }

    resposta = requests.post(
        OPENROUTER_URL,
        headers=cabecalhos,
        json=corpo_requisicao,
        timeout=180
    )

    if not resposta.ok:
        raise RuntimeError(
            f"Erro na API do OpenRouter ({resposta.status_code}): "
            f"{resposta.text}"
        )

    dados_resposta = resposta.json()

    try:
        conteudo_resposta = (
            dados_resposta["choices"][0]["message"]["content"]
        )
    except (KeyError, IndexError, TypeError) as erro:
        raise RuntimeError(
            "A resposta do OpenRouter não contém o formato esperado."
        ) from erro

    if isinstance(conteudo_resposta, dict):
        metadados = conteudo_resposta
    else:
        conteudo_resposta = limpar_bloco_json(conteudo_resposta)
        metadados = json.loads(conteudo_resposta)

    return validar_metadados(metadados)


def extrair_metadados_arquivo(
    caminho_markdown: Path
) -> dict[str, Any]:
    """
    Recebe o caminho de um arquivo .md e extrai seus metadados.
    """
    conteudo = caminho_markdown.read_text(
        encoding="utf-8",
        errors="replace"
    )

    return extrair_metadados(conteudo)


def salvar_json(
    metadados: dict[str, Any],
    caminho_saida: Path
) -> None:
    """
    Salva os metadados em um arquivo JSON.
    """
    caminho_saida.write_text(
        json.dumps(
            metadados,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def processar_arquivos_markdown() -> None:
    """
    Processa todos os arquivos .md encontrados na pasta aula_2.
    """
    if not PASTA_MARKDOWN.exists():
        print(
            f"A pasta '{PASTA_MARKDOWN}' não foi encontrada."
        )
        return

    arquivos_markdown = sorted(PASTA_MARKDOWN.glob("*.md"))

    if not arquivos_markdown:
        print(
            f"Nenhum arquivo .md foi encontrado em "
            f"'{PASTA_MARKDOWN.resolve()}'."
        )
        return

    print(
        f"Foram encontrados {len(arquivos_markdown)} "
        f"arquivos Markdown.\n"
    )

    sucessos = 0
    erros = 0

    for arquivo_markdown in arquivos_markdown:
        try:
            print(f"Processando: {arquivo_markdown.name}")

            metadados = extrair_metadados_arquivo(
                arquivo_markdown
            )

            # Exemplo:
            # bioetica_e_ia.md
            # output_bioetica_e_ia.json
            nome_saida = (
                f"output_{arquivo_markdown.stem}.json"
            )

            caminho_saida = (
                PASTA_MARKDOWN / nome_saida
            )

            salvar_json(
                metadados,
                caminho_saida
            )

            print(f"JSON salvo em: {caminho_saida}")
            print(
                json.dumps(
                    metadados,
                    ensure_ascii=False,
                    indent=2
                )
            )
            print()

            sucessos += 1

        except requests.Timeout:
            print(
                f"Erro: o tempo de resposta foi excedido "
                f"para {arquivo_markdown.name}.\n"
            )
            erros += 1

        except requests.RequestException as erro:
            print(
                f"Erro de conexão em "
                f"{arquivo_markdown.name}: {erro}\n"
            )
            erros += 1

        except Exception as erro:
            print(
                f"Erro ao processar "
                f"{arquivo_markdown.name}: {erro}\n"
            )
            erros += 1

    print("Processamento finalizado.")
    print(f"Arquivos processados: {sucessos}")
    print(f"Arquivos com erro: {erros}")


if __name__ == "__main__":
    processar_arquivos_markdown()