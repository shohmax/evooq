import click
import httpx
import os
import glob

API_URL = "http://localhost:8000"

@click.group()
def cli():
    pass

@click.command()
@click.argument('folder_path', type=click.Path(exists=True))
def upload(folder_path):

    pdf_files = glob.glob(os.path.join(folder_path, '**', '*.pdf'), recursive=True)
    
    if not pdf_files:
        click.echo("No PDF files found in the folder.")
        return

    if len(pdf_files) > 100:
        click.echo("Error: You can upload a maximum of 100 PDF files.")
        return

    files = []
    for pdf_path in pdf_files:
        files.append(('files', (os.path.basename(pdf_path), open(pdf_path, 'rb'), 'application/pdf')))
    
    try:
        response = httpx.post(f"{API_URL}/upload/", files=files)
        response.raise_for_status()
        click.echo(response.json())
    except httpx.HTTPStatusError as e:
        click.echo(f"Error: {e.response.json()['detail']}")

@click.command()
@click.argument('query')
def query(query):
    try:
        response = httpx.post(f"{API_URL}/query", data={"query": query})
        response.raise_for_status()
        click.echo(response.json()['reply'])
    except httpx.HTTPStatusError as e:
        click.echo(f"Error: {e.response.json()['detail']}")

cli.add_command(upload)
cli.add_command(query)

if __name__ == '__main__':
    cli()
