import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
from docling.document_converter import DocumentConverter


# Pasta onde estão os PDFs
PASTA_PDFS = Path("Aula2")

# Pasta onde os arquivos Markdown serão salvos
PASTA_SAIDA = Path("aula_2")


def converter_pdfs():
    # Cria a pasta aula_2 caso ela ainda não exista
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    # Procura todos os arquivos PDF dentro da pasta Aula2
    arquivos_pdf = list(PASTA_PDFS.glob("*.pdf"))

    if not arquivos_pdf:
        print(f"Nenhum PDF encontrado na pasta: {PASTA_PDFS.resolve()}")
        return

    converter = DocumentConverter()

    print(f"Foram encontrados {len(arquivos_pdf)} arquivos PDF.\n")

    for arquivo_pdf in arquivos_pdf:
        try:
            print(f"Convertendo: {arquivo_pdf.name}")

            resultado = converter.convert(str(arquivo_pdf))

            # Mantém o mesmo nome e altera apenas a extensão
            arquivo_saida = PASTA_SAIDA / f"{arquivo_pdf.stem}.md"

            conteudo_markdown = resultado.document.export_to_markdown()

            arquivo_saida.write_text(
                conteudo_markdown,
                encoding="utf-8"
            )

            print(f"Salvo em: {arquivo_saida}\n")

        except Exception as erro:
            print(f"Erro ao converter {arquivo_pdf.name}: {erro}\n")

    print("Conversão finalizada.")


if __name__ == "__main__":
    converter_pdfs()