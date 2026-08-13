import os

# ============================================================
# CORREÇÃO PYTORCH / DOCLING NO WINDOWS
# ============================================================
#
# O Docling utiliza modelos PyTorch para analisar o layout
# dos PDFs. Em alguns ambientes Windows, o PyTorch tenta
# utilizar o TorchInductor e procurar o compilador C++ cl.exe.
#
# Como esse compilador não está instalado neste computador,
# configuramos o PyTorch para continuar em modo eager caso
# a compilação otimizada falhe.
#
# IMPORTANTE:
# esta configuração deve acontecer ANTES de importar o Docling.
# ============================================================

import torch

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

except Exception:
    pass


# Evita que chamadas a torch.compile tentem utilizar
# o compilador C++ no Windows.
if hasattr(torch, "compile"):

    def compile_desativado(model, *args, **kwargs):
        return model

    torch.compile = compile_desativado


# ============================================================
# IMPORTAÇÕES
# ============================================================

from pathlib import Path

from docling.document_converter import DocumentConverter


# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent

PASTA_PDFS = PASTA_ATUAL / "pdfs"

PASTA_RESULTS = PASTA_ATUAL / "results"


# ============================================================
# FUNÇÃO AUXILIAR PARA NORMALIZAR O ID DO DOCUMENTO
# ============================================================

def gerar_document_id(pdf_path: Path) -> str:
    """
    Gera um identificador simples para cada documento.

    Exemplo:

    attention_is_all_you_need(1).pdf

    torna-se:

    attention_is_all_you_need
    """

    document_id = pdf_path.stem

    document_id = (
        document_id
        .replace("(1)", "")
        .replace(" ", "_")
        .strip("_")
    )

    return document_id


# ============================================================
# CONVERSÃO PDF -> MARKDOWN
# ============================================================

def converter_pdfs_para_markdown():
    """
    Converte automaticamente todos os arquivos PDF
    presentes em aula_04/pdfs para Markdown utilizando
    o Docling.

    Cada documento será salvo em:

    aula_04/
        results/
            nome_documento/
                markdown/
                    nome_documento.md
    """

    print("\n")
    print("=" * 70)
    print("ETAPA 1 - CONVERSÃO PDF -> MARKDOWN")
    print("=" * 70)

    # --------------------------------------------------------
    # GARANTIR EXISTÊNCIA DAS PASTAS
    # --------------------------------------------------------

    PASTA_PDFS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOCALIZAR OS PDFs
    # --------------------------------------------------------

    arquivos_pdf = sorted(
        PASTA_PDFS.glob("*.pdf")
    )

    if not arquivos_pdf:

        raise FileNotFoundError(
            "\nNenhum arquivo PDF foi encontrado.\n"
            f"Coloque os documentos em:\n{PASTA_PDFS}"
        )

    print(
        f"\nForam encontrados "
        f"{len(arquivos_pdf)} PDFs.\n"
    )

    # --------------------------------------------------------
    # INICIALIZAR DOCLING
    # --------------------------------------------------------

    print(
        "Inicializando o DocumentConverter..."
    )

    converter = DocumentConverter()

    print(
        "DocumentConverter inicializado.\n"
    )

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    documentos_convertidos = []

    documentos_com_erro = []

    documentos_existentes = []

    # --------------------------------------------------------
    # PROCESSAR TODOS OS PDFs
    # --------------------------------------------------------

    for numero, pdf_path in enumerate(
        arquivos_pdf,
        start=1
    ):

        print("\n")
        print("-" * 70)

        print(
            f"[{numero}/{len(arquivos_pdf)}] "
            f"Documento: {pdf_path.name}"
        )

        # ----------------------------------------------------
        # IDENTIFICADOR DO DOCUMENTO
        # ----------------------------------------------------

        document_id = gerar_document_id(
            pdf_path
        )

        # ----------------------------------------------------
        # PASTAS DO DOCUMENTO
        # ----------------------------------------------------

        pasta_documento = (
            PASTA_RESULTS
            / document_id
        )

        pasta_markdown = (
            pasta_documento
            / "markdown"
        )

        pasta_markdown.mkdir(
            parents=True,
            exist_ok=True
        )

        arquivo_markdown = (
            pasta_markdown
            / f"{document_id}.md"
        )

        # ----------------------------------------------------
        # EVITAR REPROCESSAMENTO
        # ----------------------------------------------------
        #
        # Se o Markdown já foi criado corretamente em uma
        # execução anterior, não gastamos tempo convertendo
        # novamente.
        # ----------------------------------------------------

        if (
            arquivo_markdown.exists()
            and arquivo_markdown.stat().st_size > 0
        ):

            markdown_existente = (
                arquivo_markdown.read_text(
                    encoding="utf-8"
                )
            )

            print(
                "Markdown já existe. "
                "Conversão ignorada."
            )

            print(
                f"Arquivo: {arquivo_markdown}"
            )

            print(
                f"Caracteres: "
                f"{len(markdown_existente):,}"
            )

            documentos_existentes.append({
                "document_id": document_id,
                "pdf": pdf_path.name,
                "markdown": str(
                    arquivo_markdown
                ),
                "characters": len(
                    markdown_existente
                )
            })

            continue

        # ----------------------------------------------------
        # TENTAR CONVERSÃO
        # ----------------------------------------------------

        try:

            print(
                "Convertendo PDF..."
            )

            resultado = converter.convert(
                pdf_path
            )

            # ------------------------------------------------
            # DOCUMENTO ESTRUTURADO DO DOCLING
            # ------------------------------------------------

            documento = resultado.document

            # ------------------------------------------------
            # EXPORTAÇÃO PARA MARKDOWN
            # ------------------------------------------------

            print(
                "Exportando para Markdown..."
            )

            markdown = (
                documento.export_to_markdown()
            )

            # ------------------------------------------------
            # VALIDAÇÃO
            # ------------------------------------------------

            if not markdown.strip():

                raise ValueError(
                    "O Markdown gerado está vazio."
                )

            # ------------------------------------------------
            # SALVAR MARKDOWN
            # ------------------------------------------------

            arquivo_markdown.write_text(
                markdown,
                encoding="utf-8"
            )

            # ------------------------------------------------
            # RESULTADO
            # ------------------------------------------------

            print(
                "STATUS: OK"
            )

            print(
                f"Markdown salvo em:\n"
                f"{arquivo_markdown}"
            )

            print(
                f"Caracteres extraídos: "
                f"{len(markdown):,}"
            )

            documentos_convertidos.append({
                "document_id": document_id,
                "pdf": pdf_path.name,
                "markdown": str(
                    arquivo_markdown
                ),
                "characters": len(markdown)
            })

        # ----------------------------------------------------
        # TRATAMENTO DE ERRO
        # ----------------------------------------------------

        except Exception as erro:

            print(
                "STATUS: ERRO"
            )

            print(
                f"Não foi possível converter "
                f"{pdf_path.name}"
            )

            print(
                f"Tipo do erro: "
                f"{type(erro).__name__}"
            )

            print(
                f"Mensagem: {erro}"
            )

            documentos_com_erro.append({
                "document_id": document_id,
                "pdf": pdf_path.name,
                "erro": str(erro)
            })

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESUMO DA CONVERSÃO")
    print("=" * 70)

    total_pdfs = len(
        arquivos_pdf
    )

    total_convertidos = len(
        documentos_convertidos
    )

    total_existentes = len(
        documentos_existentes
    )

    total_erros = len(
        documentos_com_erro
    )

    total_sucesso = (
        total_convertidos
        + total_existentes
    )

    print(
        f"\nPDFs encontrados: "
        f"{total_pdfs}"
    )

    print(
        f"Convertidos nesta execução: "
        f"{total_convertidos}"
    )

    print(
        f"Markdown já existente: "
        f"{total_existentes}"
    )

    print(
        f"Total disponível com sucesso: "
        f"{total_sucesso}"
    )

    print(
        f"Erros: "
        f"{total_erros}"
    )

    # --------------------------------------------------------
    # DOCUMENTOS CONVERTIDOS
    # --------------------------------------------------------

    if documentos_convertidos:

        print(
            "\nDOCUMENTOS CONVERTIDOS "
            "NESTA EXECUÇÃO:"
        )

        for documento in documentos_convertidos:

            print(
                f"- "
                f"{documento['document_id']} "
                f"({documento['characters']:,} "
                f"caracteres)"
            )

    # --------------------------------------------------------
    # DOCUMENTOS JÁ EXISTENTES
    # --------------------------------------------------------

    if documentos_existentes:

        print(
            "\nDOCUMENTOS QUE JÁ "
            "ESTAVAM CONVERTIDOS:"
        )

        for documento in documentos_existentes:

            print(
                f"- "
                f"{documento['document_id']} "
                f"({documento['characters']:,} "
                f"caracteres)"
            )

    # --------------------------------------------------------
    # DOCUMENTOS COM ERRO
    # --------------------------------------------------------

    if documentos_com_erro:

        print(
            "\nDOCUMENTOS COM ERRO:"
        )

        for documento in documentos_com_erro:

            print(
                f"- {documento['pdf']}"
            )

            print(
                f"  Erro: "
                f"{documento['erro'][:300]}"
            )

    # --------------------------------------------------------
    # RETORNO
    # --------------------------------------------------------

    todos_documentos_validos = (
        documentos_convertidos
        + documentos_existentes
    )

    return (
        todos_documentos_validos,
        documentos_com_erro
    )


# ============================================================
# VERIFICAR ESTRUTURA DOS MARKDOWNS
# ============================================================

def analisar_markdowns(
    documentos
):
    """
    Faz uma primeira análise simples dos Markdown gerados.

    Essa etapa será útil posteriormente no relatório para
    verificar a presença de headings, tabelas e referências
    a imagens.
    """

    print("\n")
    print("=" * 70)
    print("ANÁLISE INICIAL DOS MARKDOWNS")
    print("=" * 70)

    resultados = []

    for documento in documentos:

        caminho = Path(
            documento["markdown"]
        )

        texto = caminho.read_text(
            encoding="utf-8"
        )

        linhas = texto.splitlines()

        # ----------------------------------------------------
        # HEADINGS
        # ----------------------------------------------------

        headings = [
            linha
            for linha in linhas
            if linha.lstrip().startswith("#")
        ]

        # ----------------------------------------------------
        # POSSÍVEIS LINHAS DE TABELA MARKDOWN
        # ----------------------------------------------------

        linhas_tabela = [
            linha
            for linha in linhas
            if (
                linha.strip().startswith("|")
                and linha.strip().endswith("|")
            )
        ]

        # ----------------------------------------------------
        # REFERÊNCIAS A IMAGENS MARKDOWN
        # ----------------------------------------------------

        imagens = [
            linha
            for linha in linhas
            if "![" in linha
        ]

        resultado = {
            "document_id":
                documento["document_id"],

            "characters":
                len(texto),

            "lines":
                len(linhas),

            "headings":
                len(headings),

            "table_lines":
                len(linhas_tabela),

            "image_references":
                len(imagens)
        }

        resultados.append(
            resultado
        )

        print("\n" + "-" * 70)

        print(
            f"Documento: "
            f"{documento['document_id']}"
        )

        print(
            f"Caracteres: "
            f"{len(texto):,}"
        )

        print(
            f"Linhas: "
            f"{len(linhas):,}"
        )

        print(
            f"Headings: "
            f"{len(headings):,}"
        )

        print(
            f"Linhas de tabela: "
            f"{len(linhas_tabela):,}"
        )

        print(
            f"Referências a imagens: "
            f"{len(imagens):,}"
        )

    return resultados


# ============================================================
# MOSTRAR ESTRUTURA GERADA
# ============================================================

def mostrar_estrutura(
    documentos
):
    """
    Exibe os caminhos dos arquivos Markdown criados.
    """

    print("\n")
    print("=" * 70)
    print("ARQUIVOS MARKDOWN DISPONÍVEIS")
    print("=" * 70)

    for documento in documentos:

        print(
            f"\n{documento['document_id']}"
        )

        print(
            f"└── {documento['markdown']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - "
        "AVALIAÇÃO DE ESTRATÉGIAS DE CHUNKING"
    )
    print("=" * 70)

    print(
        "\nEtapa atual:"
    )

    print(
        "PDF -> Markdown"
    )

    # ========================================================
    # ETAPA 1
    # ========================================================

    documentos, erros = (
        converter_pdfs_para_markdown()
    )

    # ========================================================
    # SÓ CONTINUA SE TIVERMOS DOCUMENTOS
    # ========================================================

    if documentos:

        analisar_markdowns(
            documentos
        )

        mostrar_estrutura(
            documentos
        )

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print("\n")
    print("=" * 70)

    if erros:

        print(
            "ETAPA 1 FINALIZADA COM "
            f"{len(erros)} ERRO(S)"
        )

        print(
            "\nOs documentos convertidos "
            "foram preservados."
        )

        print(
            "Os documentos com erro podem "
            "ser executados novamente."
        )

    else:

        print(
            "ETAPA 1 FINALIZADA COM SUCESSO"
        )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()