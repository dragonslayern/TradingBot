<?php 

$text = file_get_contents("https://api.coinmarketcap.com/v1/ticker/?limit=10");

$objects = json_decode($text);








?>