from haystack import Pipeline

from gaz.setup import add_gaz_search_components
from lownergaz.setup import add_lownergaz_search_components
from sent.setup import add_sent_search_components

add_gaz_search_components(Pipeline(), "bm25", 10)
add_gaz_search_components(Pipeline(), "ann", 10)

add_lownergaz_search_components(Pipeline(), "bm25", 10, reset=True)
# add_lownergaz_search_components(Pipeline(), "ann", 10)

# add_sent_search_components(Pipeline(), "bm25", 10)
# add_sent_search_components(Pipeline(), "ann", 10)