# Retrieve the texts from the url links
import requests
from bs4 import BeautifulSoup
import re


# Set the end of the main content of the articles. Delete the ads at the end of the news site.
def remove_siblings_after_tag(soup, tag_name, text):
    try:
        # Find the specified tag
        tag = soup.find(tag_name, text=text)
        if tag:
            # Get the parent tag
            parent = tag.parent.parent
            for sibling in parent.find_next_siblings():
                sibling.extract()
            parent.extract()
    except Exception as e:
        print(f"Error removing siblings after tag: {e}")


# Use keywords to find the whole sentence in the webpage
def find_sentences_with_text(soup, text):
    try:
        # Use regular expressions to build patterns that contain the specified text
        pattern = re.compile(r'\b{}\b'.format(re.escape(text)), re.IGNORECASE)

        # Use the find_all() method to find a label that contains the specified text
        matching_sentences = soup.find_all(text=pattern)

        return matching_sentences
    except Exception as e:

        raise Exception(f"Error finding sentences with text: {e}")


# Retrieve the article text from url
def get_article_text_from_url(url):
    try:
        # Send a GET request to get web content
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors (e.g., 404 or 500)


        # Parse web content using Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find and extract the text content of news articles
        # Delete unwanted information
        del_tags = soup.find_all('p', class_='content-media__description') + soup.find_all(
            'figcaption') + soup.find_all('div', class_='show-multicontent-playlist')
        for del_tag in del_tags:
            del_tag.extract()

        del_texts = ['g1 no YouTube', 'Reveja os telejornais do Acre']
        del_texts += find_sentences_with_text(soup, 'VÍDEOS: ')
        for del_text in del_texts:
            remove_siblings_after_tag(soup, 'p', del_text)

        # Start in a new line for each paragraph/bullet/section
        subtitle = soup.find('h2').get_text().strip()
        lines = [subtitle]

        for element in soup.article.descendants:
            if element.name == 'p':
                if element.find(['ul', 'ol']):
                    pass
                else:
                    lines.append(element.get_text(strip=True))
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li', recursive=False):
                    lines.append(li.get_text(strip=True))
                    li.extract()
        article_text = '\n'.join(lines)
        return article_text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error for URL {url}: {e}")
    except Exception as e:
        try:
            del_tags = soup.find_all('div', class_='foto componente_materia midia-largura-620')
            for del_tag in del_tags:
                del_tag.extract()
            article_content_element = soup.find('div', class_='materia-conteudo entry-content clearfix',
                                                id='materia-letra').get_text().strip()
            return article_content_element

        except Exception as e:

            raise Exception(f"Error while processing the URL {url}: {e}")
