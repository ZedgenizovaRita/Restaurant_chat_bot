# -*- coding: utf-8 -*-
"""
orchestrator.py — агентный цикл бота

Один ход:
  1. Реплика гостя добавляется в историю диалога (формат датасета: «User: … -> Bot:»).
  2. Модель пишет ответ: текст и/или команду ACTION (protocol.py).
  3. Код проверяет только форму: ответ не оборван, команда существует и с верными параметрами,
     модель уверена (средняя log-вероятность). Иначе — честное «не понял».
  4. Команды с результатом (инструменты брони) код выполняет и возвращает модели «Tool: RESULT: …»,
     модель продолжает. Конечные команды (меню, фильтр, цена, бизнес-ланчи, notify_manager)
     код выполняет и показывает гостю.

Инструменты:
  call_date_agent(EXPRESSION)                 -> ДД.ММ.ГГГГ               «завтра», «в пятницу» -> дата
  call_booking_time_agent(DATE,TIME,…)        -> time_valid | invalid_time
  call_free_tables_agent(DATE,TIME,GUESTS,…)  -> available | unavailable | active_booking_exists
  book_table(DATE,TIME,GUESTS,NAME)           -> booked | table_busy      база заново проверяет всё сама
  check_active_booking / cancel_booking       -> бронь гостя / cancelled
  show_menu, call_menu_filter_agent, call_price_agent, show_business_lunch — данные только из menu.json

"""
from __future__ import annotations

import asyncio
import collections
import difflib
import unicodedata
import html as html_lib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from dates import moscow_now, resolve_date, validate_booking_date
from db import BookingDB, check_slot
from llm import GenResult, PromptTooLong
from menu import Menu, find_items, norm as menu_norm, fmt_price, get_menu, tool_filter, tool_price, tool_show_business_lunch, tool_show_menu
import topic_gate as tg
from menu_router import BEST_RE, COMPOSE_RE, MEAT_Q_RE, PRICE_RE, REC_RE, SPICY_Q_RE, _category_of, _list_message, _regular, _dish_hits, _same, _tokens, detect_filter, filter_messages, note_shown, route
from slots import extract as extract_slots
from protocol import DialogHistory, Reply, TOOLS_WITH_RESULT, expects_answer, parse_reply, sanitize_user_text, validate_action

log = logging.getLogger("yuqi.orchestrator")
BOOKING_TOOLS = {"call_date_agent", "call_booking_time_agent", "call_free_tables_agent", "book_table"}

FALLBACK_TEXT = ("Извините, я вас не понял. Я могу рассказать про меню, цены, бизнес-ланчи, адрес и часы работы, "
                 "а также забронировать столик. Попробуйте, пожалуйста, переформулировать.")
BAD_DATE_TEXT = "Не удалось понять дату. Напишите её, пожалуйста, например: 25 октября."    # так же в датасете
BOOKING_WORDS = re.compile(r"брон|стол|запи[сш]|посидеть|резерв")
FIELDS = (("DATE", "date", "на какую дату", "дата"), ("TIME", "time", "на какое время", "время"),
          ("GUESTS", "guests", "сколько человек будет", "гостей"), ("NAME", "name", "на чьё имя оформить бронь", "имя"))
NEED = {"call_booking_time_agent": ("DATE", "TIME"), "call_free_tables_agent": ("DATE", "TIME", "GUESTS"),
        "book_table": ("DATE", "TIME", "GUESTS", "NAME")}
CLAIM_BOOKED = re.compile(r"бронь (?:подтвержден|оформлен|создан)|столик забронирован|забронировал[аи]? (?:для вас|столик)")
CLAIM_CANCELLED = re.compile(r"отмен(?:ил|яю|яем|ю|им|ена|ено|ен|ён)\w{0,12}.{0,30}брон|брон\w{0,12}.{0,30}(?:отмен(?:ен|ён|ил|яю|ю|им)\w{0,12}|снят\w{0,12}|удал[её]н\w{0,12})")
CLAIM_EXISTS = re.compile(r"у (?:вас|нас) (?:уже )?(?:есть|забронирован\w{0,12}) (?:активн\w{1,16} )?(?:брон\w{0,12}|стол\w{0,12})|ваш\w{0,12} (?:активн\w{1,16} )?брон\w{1,16} (?:на |:|—)")
TIME_SAID = re.compile(r"\d{1,2}:\d{2}|(?<![\w])(?:в|к|на|до)\s+\d{1,2}(?![\d:.\-/])(?!\s*(?:человек|чел|гост|персон|янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек))"
                       r"|утра|вечера|часов|час(?:а|у)?\b|полдень|полноч")
OK_LEAD = {"готово", "отлично", "конечно", "хорошо", "спасибо", "супер", "замечательно", "прекрасно", "всё", "все", "итак",
           "принято", "понял", "записал", "увы", "к", "извините", "простите", "да", "нет", "сейчас", "секунду", "минутку"}
ABORT_BOOKING = re.compile(r"(?:(?<!\w)не (?:надо|нуж\w{1,16}|хочу|буду|планирую|собираюсь)|передумал\w{0,12})(?:\s+\w{1,16})?\s+(?:брон|стол|запис|резерв)"
                           r"|(?:брон\w{0,12}|стол\w{0,12})(?:\s+\w{1,16}){0,3}?\s+(?<!\w)не\s+(?:надо|нуж\w{1,16})")
ABORT_SHORT = re.compile(r"(?:(?:ой|ну|да|нет|ладно|все|я|уже)\s+)*(?:не надо|не нужно|не хочу|передумал\w*|отбой|стоп|забудь|забей|не важно|неважно)"
                         r"(?:\s+(?:уже|больше|спасибо|пожалуйста))*")      # на входе — текст без знаков препинания и «ё»
# «У вас есть мидии?», «А с рыбой?», «Какой у вас кофе?» — наличие блюда решает код по menu.json, а не модель
AVAIL = (re.compile(r"\bесть\b(?:\s+ли)?(?:\s+у\s+вас)?\s+(.+)$"),
         re.compile(r"^(?:а\s+)?(?:(?:что|какие|какой|какая|какое|блюда|блюдо|еще)\s+)*(?:с|со|из)\s+(.+)$"),
         re.compile(r"^(?:а\s+)?(?:какой|какая|какое|какие)\s+(?:у\s+вас\s+)?(.+?)(?:\s+у\s+вас)?$"),
         re.compile(r"^(?:а\s+)?(?:у\s+вас\s+)?(\w+(?:\s+\w+){0,4})\s+(?:есть|бывает|имеется)(?:\s+у\s+вас)?$"))
AVAIL_STOP = {"вас", "есть", "что", "какой", "какая", "какое", "какие", "еще", "нибудь", "можно", "пожалуйста", "для", "нас", "мне", "вот"}
# не блюдо (услуги, цены, часы) — вопрос не про наличие блюда: отвечает модель
AVAIL_SKIP = ("розетк", "чаев", "стульчик", "гардероб", "кальян", "курит", "собак", "животн", "питомц", "банкет", "мероприят", "праздник", "караоке", "сцен", "тихо", "шум", "бизнес", "ланч", "обед", "доставк", "парков", "стол", "мест", "вайфай", "wifi", "туалет", "детск", "веранд", "террас",
              "скидк", "акци", "оплат", "налич", "карт", "самовывоз", "музык", "зал", "брон", "свобод", "цен", "стоим", "дешев", "дорог",
              "работ", "час", "адрес", "телефон", "меню", "порци", "размер", "калор", "состав", "аллерг", "глютен", "лактоз",
              "мног", "выбор", "вариант", "новин", "популярн", "рекоменд", "хит", "рейтинг", "отзыв", "оценк", "сайт", "соцсет",
              "инстаграм", "телеграм", "владел", "основан", "истори", "повар", "шеф", "кухн", "ресторан", "персонал", "официант",
              "график", "расписан", "сотрудник")
# общие слова-категории: в поиске не участвуют («ролл Калифорния» -> ищем «калифор»); если кроме них ничего нет — «какие есть роллы?» -> show_menu
AVAIL_GENERIC = ("блюд", "еда", "еду", "напит", "десерт", "закуск", "горяч", "основн", "суп", "салат", "ролл", "суш", "лапш", "остр",
                 "сладк", "вегет", "веган", "постн")
AVAIL_MAX = 5
CHECK_BOOKING2 = re.compile(r"(?:мо[яюиеё]|моих)\s+(?:\w{1,16}\s+){0,2}брон\w{0,12}|у меня\s+(?:\w{1,16}\s+){0,4}брон\w{0,12}|(?:^|\s)есть\s+(?:ли\s+)?(?:активн\w{1,16}\s+)?брон\w{0,12}$")
NOT_CHECK = re.compile(r"вопрос|отмен|перенес|перенос|измен|удал|заброн|хочу|нужн|сдела|оформ|нов[уы]ю|нов[ыо]е|свободн|\d")
WORD_NUMS = {"один": 1, "одна": 1, "одного": 1, "одну": 1, "двое": 2, "два": 2, "две": 2, "двоих": 2, "вдвоем": 2, "трое": 3, "три": 3, "троих": 3,
             "втроем": 3, "четверо": 4, "четыре": 4, "четверых": 4, "вчетвером": 4, "пятеро": 5, "пять": 5, "шестеро": 6, "шесть": 6, "семеро": 7, "семь": 7}
BUSY_CLAIM = re.compile(r"занят|нет свободн|свободных (?:столик|мест)\w* нет|нет мест|не нашлось (?:свободн|столик)|мест нет")
ASKS = {"date": re.compile(r"какую дату|какой день|на какое число"), "time": re.compile(r"какое (?:точное )?время|во сколько"),
        "guests": re.compile(r"сколько (?:человек|гостей|вас)"), "name": re.compile(r"чьё имя|чье имя|какое имя|как вас зовут")}
CANCEL_REQ = re.compile(r"отмен\w{0,12}\s+(?:\w{1,16}\s+){0,2}брон|брон\w{0,12}\s+(?:\w{1,16}\s+){0,2}отмен|(?:сними|снять|удали|удалить)\w{0,12}\s+(?:\w{1,16}\s+){0,2}брон"
                        r"|не приду|не смогу (?:прийти|придти)|откаж\w{1,16} от (?:брони|стола)")
AFFIRM = re.compile(r"(?:^|\s)(?:да|давай|давайте|конечно|ага|угу|ок|окей|подтверждаю|отмен\w*)(?:\s|$)")
BARE_YES = ("да", "ага", "угу", "ок", "окей", "давай", "давайте", "конечно")
OFFER_BOOK = re.compile(r"(?:хотите|помогу|могу|желаете).{0,50}(?:брон|столик)|(?:брон|столик).{0,50}(?:хотите|помогу|желаете)")
BOOK_ASK = "Отлично! Подскажите, пожалуйста, на какую дату, на какое время, сколько человек будет и на чьё имя оформить бронь?"
HELP_ASK = ("Чем могу помочь? Могу рассказать про меню, цены, бизнес-ланчи, адрес и часы работы, а также забронировать столик.")
NO_INFO = re.compile(r"скидк|акци[ияюей]\b|акции|бонус|лояльн|промокод|купон|кэшбэк|кешбэк|сертификат|абонемент")
NO_INFO_TEXT = ("К сожалению, у меня нет информации о скидках, акциях и бонусных программах. "
                "Уточните, пожалуйста, у администратора ресторана. Могу подсказать по меню, ценам или помочь с бронью.")
GROUND_OK = re.compile(r"какие|какой|какая|какое|что у вас|покажи|расскажи|подскажи|что.нибудь|предлож|посовет")   # список/совет — любые блюда уместны
# просьбы, которых бот не умеет: набор под бюджет, заказ и доставка еды (он только отвечает про меню и бронирует столики)
STAFF = re.compile(r"повар|официант|бармен|сотрудник|персонал|ваканси|трудоустро|устро\w+ (?:к вам |у вас )?на работу|на работу|резюме|стажир|"
                   r"работать у вас|работу у вас|работа у вас|набираете|ищете (?:повар|официант|сотрудник)")
STAFF_TEXT = ("К сожалению, у меня нет информации о сотрудниках и вакансиях. Уточните, пожалуйста, у администратора ресторана. "
              "Могу рассказать про меню и цены или помочь забронировать столик.")
ALCOHOL_Q = re.compile(r"(?<!без)алкогол|спиртн|\bпив[оаеуы]|пивн|\bвин[оаеуы]\b|винн|саке|сакэ|соджу|виски|водк|шампанск|просекко|коктейл|коньяк|ликёр|ликер|сидр|"
                       r"\bджин\b|текил|\bром\b|мартини|крепк\w+ напит|покрепч|крепкое|пробков|со своим|свой алкогол|принести с собой|можно (?:с собой )?выпить|выпить")
ALCOHOL_BYO = re.compile(r"пробков|со своим|свой алкогол|свое (?:вино|пиво)|принести (?:с собой|свой|своё|свое)|с собой принести")
ALCOHOL_ITEM = re.compile(r"\bпив[оаеуы]|\bвин[оаеуы]\b|винн|саке|сакэ|соджу|виски|водк|шампанск|просекко|коньяк|ликёр|ликер|сидр|\bджин\b|текил|\bром\b|мартини|игрист|алкогол")
COMPLAINT_STRONG = re.compile(r"ужас|отврат|недоволь|жалов|жалоб|не понрав|груб|хам|претенз|разочар|испорти|обман|сыр\w+ мяс|тухл|отрав|безобраз|возмутит|кошмар|мерзк|нагруб|"
                              r"грязн|плохо|плохой|плохая|невкусн|хуже")
UNSUPPORTED = re.compile(r"(?:собер|собра|состав(?:ь|ьте|ить|ляй|лять)|подбер|подобр|набер|набра|сформир|скомпону|рассчита|посчита|уложи)\w* (?:\w+ ){0,3}(?:обед|ужин|завтрак|сет|набор|комплекс|заказ|меню|счет|чек|блюд|бюджет|рубл)"
                         r"|(?:оформ|сдела|прим)\w* (?:\w+ ){0,2}заказ|(?:дост(?:ав|ави)|привез)\w*|\bпринес(?:и|ите)\b|заказ\w* (?:\w+ ){0,2}(?:доставк|на дом|с собой)"
                         r"|(?:обед|ужин|завтрак|перекус|сет|набор)\w* (?:\w+ ){0,3}(?:за|на|до|в пределах|в рамках)\s*\d+\s*(?:руб|р\b|₽|тыс)"
                         r"|(?:посовет|подбер|подобр|собер|собра)\w* (?:\w+ ){0,4}(?:за|на|до|в пределах|в рамках)\s*\d+\s*(?:руб|р\b|₽|тыс)")
PRICE_CAP = re.compile(r"(?:дешевле|дешевое|недорог\w*|не дороже|до|меньше|ниже|не больше|не более|в пределах|бюджет\w*|на)\s*(?:чем\s*)?(\d{2,5})\s*(тыс\w*|руб\w*|р\b|₽)?")
PRICE_CAP_CTX = re.compile(r"блюд|что|поесть|покушать|перекусить|еда|еду|меню|позици|какие|есть|можно|недорог|дешев|закуск|суп|салат|ролл|лапш|десерт")
UNSUPPORTED_TEXT = ("К сожалению, я так не умею: не подбираю наборы под бюджет и не принимаю заказы. "
                    "Могу рассказать про меню и цены, подсказать блюда, назвать бизнес-ланчи или забронировать столик.")
OFFTOPIC_TEXT = ("К сожалению, с этим я помочь не могу. Я отвечаю на вопросы про меню, цены, бизнес-ланчи, адрес и часы работы, "
                 "а также помогаю забронировать столик.")
ADDRESS_Q = re.compile(r"адрес|где (?:вы|находит|располож|ваш)|как (?:добрат|проех|пройти|найти|доехат)|добира|доехат|доеду|ехат|проехат|метро|находит|"
                       r"располож|ориентир|как к вам|далеко|близко|где ресторан")
TRAVEL_Q = re.compile(r"сколько (?:\w+ ){0,3}(?:добира|ехат|идти|пешком)|долго (?:\w+ ){0,3}(?:ехат|добира|идти)|далеко|близко")
HOURS_Q = re.compile(r"до скольки|во сколько (?:вы )?(?:открыва|закрыва|работа)|час\w* работы|врем\w* работы|график|когда (?:вы )?(?:открыва|закрыва)|"
                     r"работаете|круглосуточ|открыты")
ADDRESS_TEXT = "Мы находимся по адресу: Москва, ул. Большая Никитская, 24/1. Работаем каждый день с 10:00 до 23:00."
TRAVEL_TEXT = " Сколько займёт дорога, зависит от того, откуда вы поедете, — построить маршрут можно в навигаторе."
HOURS_TEXT = "Мы работаем каждый день с 10:00 до 23:00. Бизнес-ланчи — по будням с 12:00 до 16:00."
DB_FAIL_TEXT = "Сейчас не получается обратиться к базе броней. Попробуйте, пожалуйста, чуть позже или обратитесь к администратору ресторана."
CAPS_WORDS = re.compile(r"что (?:ты|вы) (?:умеешь|умеете|можешь|можете)|чем (?:ты |вы )?(?:можешь|можете|поможешь|поможете)|что ты за бот|(?:^| )помощь(?: |$)")
CAPS_TEXT = ("Я бот ресторана YUQI. Могу рассказать про меню и цены, подсказать блюда, назвать бизнес-ланчи, адрес и часы работы, "
             "а также забронировать, проверить или отменить бронь столика.")
MAX_GUESTS = 6                  # больше — только через администратора, а не «столик занят»
GUEST_LIMIT_TEXT = ("К сожалению, через бота можно забронировать столик не более чем на 6 человек. Для большой компании обратитесь, "
                    "пожалуйста, к администратору ресторана. Если вас будет не больше 6 — подскажите, сколько человек будет?")
# «А у меня есть бронь?» — отвечает база, а не модель (она выдумывала бронь из слов диалога)
CHECK_BOOKING = re.compile(r"у меня (?:уже )?(?:есть )?(?:какая нибудь |активн\w{1,16} |действующ\w{1,16} )?брон|есть ли у меня (?:уже )?(?:активн\w{1,16} |действующ\w{1,16} )?брон"
                           r"|(?:какая|какой) у меня брон|на когда у меня брон|мо[яю] брон\w{1,16} (?:есть|на когда)"
                           r"|(?:проверь\w*|посмотри\w*|покажи\w*|напомни\w*) (?:мне )?(?:мо[юи] |актив\w+ )?брон")
NONFOOD = re.compile(r"жизн|смысл|любов|счасть|погод|\bмарс|\bлун[аыуе]|земл|космос|\bбог|дьявол|президент|политик|футбол|кино|фильм|сериал|музык|\bигр|деньг|школ|университет|машин|телефон|"
                     r"компьютер|интернет|работ|ваканси|собеседник|друг[а-я]* ресторан|конкурент|шанс|надежд|правд|ложь|мысл|душ[аиу]\b|сердц")
# непривычное блюдо -> что в меню похоже по формату («хинкали» ~ гёдза)
DISH_HINTS = ((re.compile(r"хинкал|манты|вареник|пельмен|равиол|вонтон|дамплинг|пян.?се"), re.compile(r"гёдз|гедз|пельмен|димсам|вонтон|\bбао\b|клёцк"), "по формату"),
              (re.compile(r"спагетти|паста|макарон|феттучин|карбонар"), re.compile(r"лапш|рамен|удон|\bсоба\b|фунчоз|\bвок\b"), "лапша"),
              (re.compile(r"стейк|бифштекс|говядин"), re.compile(r"говяд|рамен|пибимпап"), "из говядины"),
              (re.compile(r"курин|курица|шашлык|гриль"), re.compile(r"кури|карааге"), "из курицы"),
              (re.compile(r"креветк|морепрод|рыб|лосос|тунец"), re.compile(r"креветк|лосос|тунц|угор|краб|рыб"), "из морепродуктов"))
CAT_HINTS = (("drinks", "Напитки", re.compile(r"капучин|латте|эспрессо|американо|\bраф\b|мокко|какао|кофе|\bчай|матч|смузи|мохито|лимонад|морс|компот|\bвод[аыу]\b|газировк|\bкола\b|пепси|\bсок\b|напит")),
             ("desserts", "Десерты", re.compile(r"торт|чизкейк|пирожн|мороженое|макарон|круассан|эклер|панкейк|\bблин|десерт|сладк|шоколад|тирамису|пудинг|вафл|кекс|выпечк")),
             ("soups", "Супы", re.compile(r"борщ|солянк|\bуха\b|харчо|окрошк|\bсуп|бульон")),
             ("salads", "Салаты", re.compile(r"цезар|оливье|салат")))
SYNONYMS = {"лапш": ("лапш", "рамен", "ramen", "удон", "udon", "соба", "soba", "фунчоз", "вок"),
            "рыб": ("рыб", "лосос", "семг", "тунец", "тунц", "угор", "угр", "форел", "сибас", "окун", "скумбри"),   # «А с рыбой?»
            "морепрод": ("креветк", "мидии", "мидий", "кальмар", "осьминог", "гребеш", "краб", "устриц"),
            "мяс": ("говяд", "свинин", "курин", "кури", "утк", "утин", "баранин", "бекон", "стейк")}
YES = ("да", "ага", "угу", "конечно", "подтверждаю", "верно", "давай", "давайте", "ок", "окей")
KEEP_EXCHANGES = 2          # после завершённого ответа модель видит 2 прошлых обмена (если контекст ей помогает, см. _turn)
WEEKDAYS = {"понедельник": 1, "вторник": 1, "сред": 1, "четверг": 1, "пятниц": 1, "суббот": 0, "воскресен": 0, "выходн": 0, "будн": 1}
LUNCH_SCHED = "с понедельника по пятницу с 12:00 до 16:00"


_LAT2CYR = str.maketrans("aceopxykmthbACEOPXYKMTHB", "асеорхукмтнвАСЕОРХУКМТНВ")      # латинские «двойники» в русском слове: «удoн», «мeню»


def clean_unicode(text: str) -> str:
    """Невидимые и «хитрые» символы не должны ломать разбор: NFKC, без zero-width/мягких переносов, пробелы схлопнуты,
    латинские двойники внутри русских слов заменены («мeню» -> «меню»). Чисто латинские слова («Black Ramen») не трогаем."""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) not in ("Cf", "Cc") or ch in "\n\t")
    t = " ".join(t.split())
    def fix(m):
        w = m.group(0)
        return w.translate(_LAT2CYR) if re.search(r"[а-яёА-ЯЁ]", w) and re.search(r"[a-zA-Z]", w) else w
    return re.sub(r"[A-Za-zА-Яа-яЁё]+", fix, t)


def booking_line(d: str, t: str, g: int, n: str) -> str:
    w = "человека" if g % 10 in (2, 3, 4) and g % 100 not in (12, 13, 14) else "человек"
    return f"{d}, {t}, {g} {w}, имя {n}"


@dataclass
class Message:
    text: str
    html: bool = False            # True -> отправлять с parse_mode="HTML"


@dataclass
class ManagerNotice:
    """Тихое уведомление администратору (жалоба в отзыве). Гость его не видит."""
    chat_id: int
    user_name: Optional[str]
    text: str


@dataclass
class TurnResult:
    messages: list[Message] = field(default_factory=list)
    notices: list[ManagerNotice] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)   # отладка: что генерировала модель
    source: str = ""                                  # «модель», «модель + инструмент», «запасной ответ»


@dataclass
class Session:
    history: DialogHistory = field(default_factory=DialogHistory)
    last_active: Optional[datetime] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_armed: bool = False      # бот показал гостю его бронь -> cancel_booking разрешён
    cancelled: object = None        # только что отменённая бронь (для сверки слов модели)
    prev_raws: set = field(default_factory=set)      # что модель сгенерировала в прошлый ход (ловим повторы)
    prev_user: str = ""
    prev_msgs: list = field(default_factory=list)    # что бот отправил в прошлый ход
    used_ctx: bool = False
    pending_book: Optional[dict] = None   # модель хочет book_table — ждём «да» (или имя) от гостя
    slots: dict = field(default_factory=dict)       # дата/время/гости/имя, найденные КОДОМ в словах гостя
    last_slots: dict = field(default_factory=dict)  # то же только из последней реплики
    booking_ok: bool = False        # гость действительно говорит о брони (проверено один раз за разговор)
    ctx_closed: bool = False        # прошлый вопрос закрыт: контекст — только подсказка, модель может начать с чистого листа
    menu_ctx: dict = field(default_factory=dict)    # что гость только что увидел из меню («а он острый?», «давай другое», «а почему?») — для menu_router
    user_texts: list = field(default_factory=list)   # реплики гостя в текущем разговоре (проверка имени и числа гостей)

    def reset(self) -> None:
        self.history.reset()
        self.cancel_armed = False
        self.pending_book = None
        self.slots, self.last_slots = {}, {}
        self.booking_ok = False
        self.ctx_closed = False
        self.user_texts = []
        self.menu_ctx = {}

    def close_dialog(self) -> None:
        """Вопрос закрыт: в истории остаются только последние KEEP_EXCHANGES обменов (контекст для «а в выходные?»)."""
        segs, starts = self.history.done, [i for i, s in enumerate(self.history.done) if s.startswith("User:")]
        self.history.done = [] if not KEEP_EXCHANGES else segs[starts[-KEEP_EXCHANGES]:] if len(starts) >= KEEP_EXCHANGES else segs[:]
        self.ctx_closed = True
        self.slots = {}


class Orchestrator:
    def __init__(self, llm, booking_db: BookingDB, *, menu: Optional[Menu] = None, min_logprob: Optional[float] = -1.5,
                 session_ttl_min: int = 30, max_tool_steps: int = 6, now_fn: Callable[[], datetime] = moscow_now,
                 confirm_booking: bool = True, grounding: bool = True, context_choice: bool = True, use_menu_router: bool = True, topic_gate: bool = True, **_old_flags) -> None:
        """llm — объект с async agenerate(prompt) -> GenResult (llm.LLM).
        min_logprob — порог уверенности модели; ниже -> «не понял». None отключает проверку.
        confirm_booking — перед записью брони гость подтверждает данные «да».
        grounding — проверки брони кодом (имя из слов гостя, бронь только по просьбе, сверка слов модели с базой)."""
        self.llm, self.db, self.menu = llm, booking_db, menu
        self.confirm_booking, self.grounding, self.context_choice = confirm_booking, grounding, context_choice
        self.topic_gate = topic_gate                  # вопрос «не по теме» и ответ без слов-оснований в вопросе гостя отсекает код (topic_gate.py)
        self.use_menu_router = use_menu_router        # вопросы про меню сначала разбирает menu_router (код по menu.json); False — если route() уже вызывается снаружи
        self.min_logprob, self.ttl_min, self.max_tool_steps, self.now = min_logprob, session_ttl_min, max_tool_steps, now_fn
        self._sessions: dict[int, Session] = {}
        self.stats = collections.Counter()           # сколько ответов дала модель, а сколько — запасной ответ

    # ------------------------------------------------------------ сессии
    def _session(self, chat_id: int) -> Session:
        if len(self._sessions) > 2000:                                   # бот работает неделями: забытые чаты не копим в памяти
            cutoff = self.now().timestamp() - 2 * self.ttl_min * 60
            for cid in [c for c, ss in self._sessions.items()
                        if c != chat_id and not ss.lock.locked() and (ss.last_active is None or ss.last_active.timestamp() < cutoff)]:
                del self._sessions[cid]
        return self._sessions.setdefault(chat_id, Session())

    def reset(self, chat_id: int) -> None:
        """Сбросить диалог (команда /start)."""
        if chat_id in self._sessions:
            self._sessions[chat_id].reset()

    def wipe(self, chat_id: int) -> tuple[int, int]:
        """/reset для тестов: забыть диалог и стереть брони и отзывы этого чата. -> (броней, отзывов)."""
        self._sessions.pop(chat_id, None)
        return self.db.clear_chat(chat_id)

    def _menu(self) -> Menu:
        return self.menu or get_menu()

    # ------------------------------------------------------------ главный вход
    async def handle_text(self, chat_id: int, text: str, user_name: Optional[str] = None, on_progress=None) -> TurnResult:
        """on_progress — async-функция(Message): реплики модели перед инструментом («Проверяю время.») уходят гостю сразу."""
        sess = self._session(chat_id)
        async with sess.lock:                                            
            now = self.now()
            if sess.last_active and (now - sess.last_active).total_seconds() > self.ttl_min * 60:
                sess.reset()                                             
            sess.last_active = now
            clean = clean_unicode((sanitize_user_text(text) or "?")[:1500]) or "?"                  # длиннее — не вопрос гостя; заодно ограничивает время разбора
            
            # --- быстрый перехват отказа от бронирования в процессе ---
            # «не хочу бронировать» — отказ сам по себе; короткое «уже не надо», «передумал», «отбой» — только если шла бронь
            norm = self._norm(clean)
            is_abort = ABORT_BOOKING.search(norm) or (ABORT_SHORT.fullmatch(norm) and self._booking_in_progress(sess))

            if is_abort:
                sess.reset() # Сбрасываем все собранные данные
                res = TurnResult([Message("Хорошо, понял. Если понадоблюсь — обращайтесь!")], source="код: отказ от брони")
                
                # Запоминаем для истории, чтобы бот не повторялся потом
                sess.prev_msgs = [m.text for m in res.messages]
                sess.prev_raws, sess.prev_user = set(), clean.lower()
                self.stats[res.source] += 1
                return res
            # ----------------------------------------------------------------------

            sess.user_texts = (sess.user_texts + [clean])[-8:]
            # --- «есть ли блюдо X» решает код, а не модель (она выдумывает блюда, «кофе есть» без кофе и т.п.) ---
            if not sess.pending_book:
                res = await self._code_answer(chat_id, sess, clean, norm)
                if res:
                    return res
            expecting = self._expecting(sess)
            sess.last_slots = extract_slots(clean, now.date(), expecting)
            if expecting != "time" and not TIME_SAID.search(clean.lower()):
                sess.last_slots.pop("time", None)                        # «нас будет 4» — это гости, а не 16:00
            if "guests" not in sess.last_slots and self._word_guests(norm, expecting):
                sess.last_slots["guests"] = str(self._word_guests(norm, expecting))     # «Один», «нас двое», «человек один»
            if len(sess.last_slots.get("name", "")) < 3:
                sess.last_slots.pop("name", None)                        # «Не», «Да» — не имена
            g = sess.last_slots.get("guests")
            if g and int(g) > MAX_GUESTS:
                sess.last_slots.pop("guests")                            # больше 6 в слоты не попадает никогда
                if sess.pending_book or self._booking_in_progress(sess) or BOOKING_WORDS.search(norm):
                    sess.pending_book = None                             # на шаге подтверждения назвали >6 гостей: прежнюю бронь не подтверждаем
                    sess.slots.pop("guests", None)
                    sess.slots.update(sess.last_slots)                   # «12 человек» — не «столик занят», а лимит бронирования
                    return self._direct(sess, clean, GUEST_LIMIT_TEXT, "код: больше 6 гостей", close=False)
            sess.slots.update(sess.last_slots)
            if (self.grounding and not sess.pending_book and sess.last_slots and "отмен" not in norm
                    and self._slots_complete(sess) and self._booking_wanted(sess)):
                try:
                    return await self._drive_booking(chat_id, sess, clean)   # дата, время, гости и имя названы — проверяет база, а не модель
                except Exception:
                    log.exception("chat=%s: проверка брони кодом упала, дальше работает модель", chat_id)
            if sess.pending_book:
                first = norm.split()[0] if norm.split() else ""
                if first not in YES and first not in ("нет", "не", "неа") and not sess.last_slots:
                    side = await self._code_answer(chat_id, sess, clean, norm)    # «а сколько стоит удон?» посреди подтверждения
                    if side:
                        p = sess.pending_book
                        side.messages.append(Message(f"Возвращаюсь к брони: {booking_line(p['DATE'], p['TIME'], int(p['GUESTS']), p['NAME'])}. "
                                                     "Подтверждаете бронь? Напишите «да» или «нет»."))
                        return side
                res = await self._confirm(chat_id, sess, clean, user_name, on_progress)
            else:
                res = await self._turn(chat_id, sess, clean, user_name, on_progress)
            if self.grounding:
                try:
                    await self._check_facts(chat_id, sess, res)
                    await self._check_claims(chat_id, res)
                except Exception:
                    log.exception("chat=%s: сверка слов модели с базой упала", chat_id)
            texts = [m.text for m in res.messages]
            shown = [m.text for m in res.messages if m.html] or texts      # для меню сравниваем сами списки блюд
            if shown and shown == sess.prev_msgs and sess.used_ctx and clean.lower() != sess.prev_user and texts != [FALLBACK_TEXT] \
                    and not res.source.startswith("код"):                # ответ из базы повторять можно
                log.warning("chat=%s: бот повторил прошлое сообщение на другой вопрос -> «не понял»", chat_id)
                sess.reset()                                             # одно и то же дважды подряд не отправляем
                res = TurnResult([Message(FALLBACK_TEXT)], steps=res.steps, source="запасной ответ")
                shown = [FALLBACK_TEXT]
            sess.prev_msgs = shown
            sess.prev_raws, sess.prev_user = {st["raw"].strip() for st in res.steps if st.get("raw")}, clean.lower()
            self.stats[res.source] += 1
            log.info("chat=%s источник ответа: %s", chat_id, res.source)
            return res

    async def _turn(self, chat_id: int, sess: Session, user_text: str, user_name: Optional[str], on_progress) -> TurnResult:
        h, first = sess.history, None
        prompt = h.prompt_for_user(user_text)
        if self.context_choice and sess.ctx_closed and h.done:
            # Новый вопрос после закрытого: модель отвечает дважды — с прошлым контекстом и без него, — берём ответ,
            # в котором она увереннее. «А сколько стоит?» без контекста непонятен, а «Скинь меню» контекст только сбивает.
            try:
                with_ctx = await self.llm.agenerate(prompt)
                saved, h.done = h.done, []
                fresh_prompt = h.prompt_for_user(user_text)
                fresh = await self.llm.agenerate(fresh_prompt)
            except PromptTooLong:
                saved, with_ctx, fresh, h.done = [], None, None, []         # контекст слишком длинный — начинаем с чистого листа
                fresh_prompt = h.prompt_for_user(user_text)
            if with_ctx and self._score(with_ctx, sess, user_text, ctx=True) > self._score(fresh, sess, user_text):
                h.done, first = saved, with_ctx
                h.prompt_for_user(user_text)
            else:
                prompt, first = fresh_prompt, fresh
                sess.user_texts, sess.booking_ok, sess.slots = [user_text], False, dict(sess.last_slots)
        sess.ctx_closed = False
        sess.used_ctx = bool(h.done)                 # модель видела прошлые реплики (только тогда повтор — подозрителен)
        return await self._loop(chat_id, sess, user_text, user_name, on_progress, prompt, TurnResult(source="модель"), first)

    def _score(self, gen: Optional[GenResult], sess: Session, user_text: str, ctx: bool = False) -> float:
        if gen is None:
            return float("-inf")
        r = parse_reply(gen.text)
        if ctx and r.raw.strip() in sess.prev_raws and user_text.lower() != sess.prev_user:
            return float("-inf")
            
        # Блокируем выбор контекста, если модель пытается начать бронь обычным текстом без разрешения
        if not r.has_action and self.grounding and not self._booking_wanted(sess):
            if re.search(r"какую дату|какое время|сколько (человек|гостей)|чьё имя|оформить бронь|(дата|время|гостей|имя) —", r.text.lower()):
                return float("-inf")
                
        if self._problem(gen, r) or (self.grounding and r.action in BOOKING_TOOLS and not self._booking_wanted(sess)) \
                or (r.action == "cancel_booking" and not sess.cancel_armed) \
                or (r.action == "call_price_agent" and not self._item_said(sess.user_texts if ctx else [user_text], r.params["ITEM"])):
            return float("-inf")
        return gen.mean_logprob

    async def _loop(self, chat_id: int, sess: Session, user_text: str, user_name: Optional[str], on_progress,
                    prompt: str, out: TurnResult, first: Optional[GenResult] = None) -> TurnResult:
        h = sess.history
        for _ in range(self.max_tool_steps + 1):
            try:
                gen: GenResult = first or await self.llm.agenerate(prompt)
                first = None
            except PromptTooLong:
                return self._fallback(sess, out, "слишком длинный диалог")
            reply = parse_reply(gen.text)
            out.steps.append({"prompt": prompt, "raw": reply.raw, "logprob": round(gen.mean_logprob, 3), "action": reply.action})
            problem = self._problem(gen, reply)
            if problem:
                return self._fallback(sess, out, f"chat={chat_id}: {problem} | raw={reply.raw!r}")
            act_kind = {"cancel_booking": "cancel", "check_active_booking": "mybooking"}.get(reply.action)
            if act_kind and not self._justified(sess, user_text, [act_kind]):      # «Привет» -> cancel_booking: в вопросе нет ни слова о брони
                return self._offtopic(sess, out, f"chat={chat_id}: в вопросе нет слов для {reply.action}: {reply.raw!r}")

            # --- просто текст
            if not reply.has_action:
                text = reply.text
                if "Продолжим бронь" in text and not h.done:
                    text = text.split("Продолжим бронь")[0].strip() or text   # модель «вспомнила» бронь, которой нет
                text = re.sub(r"(«[^»]+»[^.«]*?) или \1", r"\1", text)        # «X или X» — модель повторила одно и то же
                text = self._lunch_day(user_text, text)
                if "не по теме" in text:
                    text = self._dish_on_menu(user_text) or re.sub(r"^Да, это", "Извините, это", text)
                
                if self.grounding:
                    is_booking_talk = re.search(r"какую дату|какое время|сколько (человек|гостей)|чьё имя|оформить бронь", text.lower())
                    has_facts = re.search(r"(дата|время|гостей|имя) — ", text.lower())
                    
                    # 1. Если модель заговорила о брони (или выдумывает факты), а гость этого не просил -> честное "не понял"
                    if (is_booking_talk or has_facts) and not self._booking_wanted(sess):
                        return self._fallback(sess, out, f"chat={chat_id}: модель начала бронь в тексте без просьбы гостя: {reply.raw!r}")
                        
                    # 1б. «Столик занят» — только если это сказала база в этом ходу; иначе модель выдумала
                    if BUSY_CLAIM.search(text.lower()) and self._booking_wanted(sess) \
                            and not any(st.get("result") in ("unavailable", "table_busy") for st in out.steps):
                        log.warning("chat=%s: модель написала «занят» без проверки базы: %r", chat_id, text)
                        if self._slots_complete(sess):
                            return await self._drive_booking(chat_id, sess, user_text)
                        return self._fallback(sess, out, f"chat={chat_id}: выдуманное «столик занят», данных брони не хватает")
                    # 1в. Модель переспрашивает то, что гость уже сказал (код это нашёл в его словах)
                    asked_known = [k for k, rx in ASKS.items() if rx.search(text.lower()) and k in sess.slots]
                    if asked_known and self._booking_wanted(sess):
                        if self._slots_complete(sess):
                            return await self._drive_booking(chat_id, sess, user_text)
                        text = self._ask_missing(sess, [f[0] for f in FIELDS if f[1] not in sess.slots]) or text
                    # 2. Если бронь запрошена, но модель галлюцинирует конкретные цифры (или пишет "Понял/Записал")
                    if (has_facts or re.match(r"(Понял|Записал|Принял):", text)) and not self._ack_ok(sess, text):
                        text = self._ask_missing(sess, [f[0] for f in FIELDS if f[1] not in sess.slots]) or text   # не верим цифрам модели
                
                if not self._justified(sess, user_text, tg.text_kinds(text)):
                    return self._offtopic(sess, out, f"chat={chat_id}: в вопросе нет слов, оправдывающих ответ модели {sorted(tg.text_kinds(text))}: {reply.raw!r}")
                out.messages.append(Message(text))
                if self.grounding and self.use_menu_router and not self._dishes_grounded(sess, user_text, text):
                    return self._fallback(sess, out, f"chat={chat_id}: модель назвала блюдо, о котором гость не спрашивал: {reply.raw!r}")
                self._note_shown(sess, None, text)
                h.commit(reply.raw)
                if not expects_answer(reply):
                    sess.close_dialog()
                return out

            if reply.action == "cancel_booking":                           # «Отменяю бронь» от модели — делает код и по базе
                confirmed = sess.cancel_armed and bool(AFFIRM.search(f" {self._norm(user_text)} "))
                text = await self._cancel_core(chat_id, sess, confirmed)
                out.messages.append(Message(text))
                out.source = "код: отмена брони"
                if text.startswith("Бронь отменена"):
                    out.steps.append({"action": "cancel_booking", "raw": reply.raw, "result": "cancelled"})
                h.commit(text)
                if "Отменить её?" not in text:
                    sess.close_dialog()
                return out
            out.source = "модель + инструмент"
            # --- инструмент с результатом: код выполняет, модель продолжает
            if self.grounding and reply.action in BOOKING_TOOLS:
                if not self._booking_wanted(sess):
                    return self._fallback(sess, out, f"chat={chat_id}: модель начала бронь без просьбы гостя: {reply.raw!r}")
                sess.booking_ok = True
                missing = self._ground(sess, reply)                      # факты брони — только из слов гостя
                if missing:
                    ask = self._ask_missing(sess, missing)
                    out.messages.append(Message(ask))
                    h.commit(ask)
                    return out
            if reply.action == "book_table" and self.confirm_booking:      # записывает бронь только после «да» гостя
                h.commit(reply.raw)
                return self._ask_confirm(sess, dict(reply.params), out)
            if reply.action in TOOLS_WITH_RESULT:
                if reply.text:                                           # «Проверяю время.» — сразу, до выполнения
                    if on_progress:
                        await on_progress(Message(reply.text))
                    else:
                        out.messages.append(Message(reply.text))
                try:
                    result, direct_text = await self._run_tool(chat_id, sess, reply)
                except Exception:
                    log.exception("chat=%s: инструмент %s упал", chat_id, reply.action)
                    return self._fallback(sess, out, "инструмент упал")
                if direct_text:                                          # код сам объясняет гостю (дата не понята и т.п.)
                    out.messages.append(Message(direct_text))
                    h.commit(direct_text)
                    if not expects_answer(Reply(raw=direct_text, text=direct_text)):
                        sess.close_dialog()
                    return out
                h.commit(reply.raw)
                out.steps[-1]["result"] = result
                prompt = h.prompt_for_tool(result)
                continue

            # --- конечное действие: меню, фильтр, цена, бизнес-ланчи, жалоба администратору
            kind = {"show_menu": "signature" if reply.params.get("CATEGORY") == "signature" else "menu", "show_business_lunch": "lunch",
                    "call_menu_filter_agent": "filter", "call_price_agent": "price", "notify_manager": "review"}.get(reply.action)
            if kind and not self._justified(sess, user_text, [kind]):
                return self._offtopic(sess, out, f"chat={chat_id}: в вопросе нет слов для {reply.action}: {reply.raw!r}")
            if reply.action == "call_menu_filter_agent" and reply.params["FILTER"] == "spicy" \
                    and re.search(r"не\s*остр", f"{user_text} {reply.text}".lower()):
                reply.params["FILTER"] = "not_spicy"                     # «А неострые?» — модель написала «Неострые…», а команду перепутала
                reply.raw = reply.raw.replace("FILTER: spicy", "FILTER: not_spicy")
            if reply.action == "call_price_agent" and not self._item_said(sess.user_texts, reply.params["ITEM"]):
                return self._fallback(sess, out, f"chat={chat_id}: блюдо {reply.params['ITEM']!r} гость не называл")
            if reply.action == "call_menu_filter_agent":
                reply.params["FILTER"] = self._spicy_filter(user_text, reply)
            if reply.text:
                out.messages.append(Message(reply.text))
            self._run_terminal(chat_id, user_name, user_text, reply, out)
            self._note_shown(sess, reply)
            h.commit(reply.raw)
            if not sess.slots:                                           # в разговоре о брони показ меню/цены не закрывает диалог и не стирает слоты
                sess.close_dialog()
            return out

        return self._fallback(sess, out, f"chat={chat_id}: слишком много шагов инструментов")

    def _fallback(self, sess: Session, out: TurnResult, why: str) -> TurnResult:
        log.warning(why)
        if self.grounding and not sess.pending_book and sess.user_texts and (sess.slots or sess.booking_ok) and self._booking_wanted(sess):
            # шла бронь: одна непонятая реплика не должна стирать уже собранные дату, время, гостей и имя — спрашиваем недостающее
            missing = [k for k, key, _, _ in FIELDS if key not in sess.slots]
            res = TurnResult(steps=out.steps, source="код: бронь")
            if missing:
                res.messages.append(Message(self._ask_missing(sess, missing)))
            else:
                self._ask_confirm(sess, {k: str(sess.slots[key]) for k, key, _, _ in FIELDS}, res)
            sess.history.reset()
            sess.history.prompt_for_user(sess.user_texts[-1])
            sess.history.commit(res.messages[0].text)
            sess.ctx_closed = False
            return res
        sess.reset()
        return TurnResult([Message(FALLBACK_TEXT)], steps=out.steps, source="запасной ответ")

    # ------------------------------------------------------------ бронь под защитой кода
    def _ask_confirm(self, sess: Session, p: dict, out: TurnResult) -> TurnResult:
        sess.pending_book = p
        out.messages.append(Message(f"Проверьте, пожалуйста: {booking_line(p['DATE'], p['TIME'], int(p['GUESTS']), p['NAME'])}. "
                                    "Подтверждаете бронь? Напишите «да» или «нет»."))
        return out

    async def _confirm(self, chat_id: int, sess: Session, text: str, user_name, on_progress) -> TurnResult:
        """Ответ на «Подтверждаете бронь?»: «да» — код пишет бронь в базу, итог по результату пишет модель."""
        p, sess.pending_book = sess.pending_book, None
        keymap = {"date": "DATE", "time": "TIME", "guests": "GUESTS", "name": "NAME"}
        changed = {k: v for k, v in sess.last_slots.items() if k in keymap and str(v) != str(p.get(keymap[k]))}
        if changed:                                                      # гость поправил данные: не стираем бронь, а проверяем и подтверждаем заново
            if self._slots_complete(sess):
                try:
                    return await self._drive_booking(chat_id, sess, text)
                except Exception:
                    log.exception("chat=%s: повторная проверка брони упала", chat_id)
            else:
                missing = [k for k, key, _, _ in FIELDS if key not in sess.slots]
                return self._direct(sess, text, self._ask_missing(sess, missing) or FALLBACK_TEXT, "код: бронь", close=False)
        words = text.lower().replace(",", " ").replace("!", " ").replace(".", " ").split()
        if not (words and words[0] in YES):                              # не подтвердил — бронь не пишем, разговор с начала
            sess.reset()
            if words and words[0] in ("нет", "не", "неа"):
                return TurnResult([Message("Хорошо, бронь не оформляю. Напишите, на какую дату, время и сколько гостей — "
                                           "и я проверю ещё раз.")], source="код: бронь")
            return await self._turn(chat_id, sess, text, user_name, on_progress)
        return await self._book(chat_id, sess, p, text, user_name, on_progress, TurnResult(source="модель + инструмент"))

    async def _book(self, chat_id: int, sess: Session, p: dict, text: str, user_name, on_progress, out: TurnResult) -> TurnResult:
        """Запись в базу (база сама заново проверяет время и столы); итог по результату пишет модель."""
        try:
            result, direct_text = await self._run_tool(chat_id, sess, Reply(raw="", action="book_table", params=p))
        except Exception:
            log.exception("chat=%s: запись брони в базу упала", chat_id)
            sess.reset()
            return TurnResult([Message("Не получилось записать бронь. Попробуйте, пожалуйста, чуть позже или обратитесь к администратору ресторана.")],
                              source="код: бронь")
        if direct_text:
            sess.reset()
            return TurnResult([Message(direct_text)], source="код: бронь")
        out.steps.append({"action": "book_table", "result": result, "raw": "(подтверждено гостем)"})
        return await self._loop(chat_id, sess, text, user_name, on_progress, sess.history.prompt_for_tool(result), out)

    @staticmethod
    def _spicy_filter(user_text: str, reply: Reply) -> str:
        """«Какие блюда НЕ острые?» — модель путает spicy и not_spicy; отрицание в словах гостя (или в её же тексте) решает код."""
        f = reply.params["FILTER"]
        if f not in ("spicy", "not_spicy"):
            return f
        t = (user_text + " " + reply.text).lower()
        neg = re.search(r"не\s*остр|без\s+остр|не\s+сильно\s+остр|мягк", t)
        return "not_spicy" if neg else "spicy" if "остр" in user_text.lower() else f

    @staticmethod
    def _item_said(texts: list, item: str) -> bool:
        """Цену какого блюда спрашивают — гость должен был назвать это блюдо (сейчас или в прошлом вопросе)."""
        said = " ".join(texts).lower().replace("ё", "е")
        words = [w for w in re.findall(r"[а-яёa-z]{3,}", item.lower().replace("ё", "е"))]
        return not words or any(w[:4] in said for w in words)

    def _justified(self, sess: Session, user_text: str, kinds) -> bool:
        """Ответ типа kind (фирменные блюда, часы, адрес, цены, меню, фильтр, бизнес-ланч, реакция на отзыв) модель выдала не из воздуха:
        в реплике гостя (для коротких «а …» — и в прошлой) есть слова, на которые такой ответ вообще бывает."""
        if not self.topic_gate or not kinds:
            return True
        now = self._norm(user_text)
        prev = self._norm(sess.user_texts[-2]) if len(sess.user_texts) >= 2 else None
        texts = tg.evidence_texts(now, prev)
        in_flow = bool(sess.slots) or (sess.booking_ok and not sess.ctx_closed)
        named = bool(_dish_hits(user_text, self._menu()) or _category_of(_tokens(user_text)))
        for k in kinds:
            if in_flow and k != "signature":
                continue                                                 # в разговоре о брони «мы работаем с 10 до 23» уместно без слов «часы»
            if tg.output_ok(k, texts) or (named and k in ("signature", "menu", "price", "filter")):
                continue
            return False
        return True

    def _offtopic(self, sess: Session, out: TurnResult, why: str) -> TurnResult:
        log.warning(why)
        if sess.slots or (sess.booking_ok and not sess.ctx_closed):
            return self._fallback(sess, out, why)                       # шла бронь — собранное не стираем
        sess.reset()
        return TurnResult([Message(OFFTOPIC_TEXT)], steps=out.steps, source="запасной ответ")

    def _dishes_grounded(self, sess: Session, user_text: str, text: str) -> bool:
        """Блюда в «ёлочках» из ответа модели: гость их назвал, они были только что показаны, или гость просил совет/список.
        Иначе модель подсовывает «любимое» блюдо (в датасете это YUQI Dragon Roll) невпопад — лучше честное «не понял»."""
        try:
            named = {menu_norm(x) for x in re.findall(r"«([^»]+)»", text)}
            if not named or self._booking_in_progress(sess):
                return True
            t = menu_norm(user_text)
            if REC_RE.search(t) or GROUND_OK.search(t) or BEST_RE.search(t) or detect_filter(user_text):
                return True
            follow = len(tg.evidence_texts(t, "x")) > 1                          # короткое «а …» / «он…» — продолжение прошлого ответа
            shown, q = (set(sess.menu_ctx.get("names") or ()) if follow else set()), [w for w in _tokens(user_text) if len(w) >= 3]
            for it in self._menu().items:
                names = [it.name, *it.aliases]
                if not any(menu_norm(n) in named or any(menu_norm(n) in x for x in named) for n in names):
                    continue
                if it.name in shown or any(_same(a, b) for a in q for n in names for b in _tokens(n) if len(b) >= 3):
                    continue
                return False
            return True
        except Exception:
            log.exception("не удалось проверить блюда в ответе модели")
            return True

    def _note_shown(self, sess: Session, reply: Optional[Reply], text: str = "") -> None:
        """Запоминаем для menu_router, что гость увидел (меню, фильтр, цена, блюда в ответе модели): «а он острый?», «давай другое»."""
        if not self.use_menu_router:
            return
        try:
            m, p = self._menu(), reply.params if reply else {}
            a = reply.action if reply else None
            note_shown(sess.menu_ctx, m, category=p.get("CATEGORY") if a == "show_menu" else None,
                       flt=p.get("FILTER") if a == "call_menu_filter_agent" else None,
                       item=p.get("ITEM") if a == "call_price_agent" else None, lunch=a == "show_business_lunch", text=text)
        except Exception:
            log.exception("не удалось запомнить показанное меню")

    @staticmethod
    def _word_guests(norm: str, expecting: Optional[str]) -> Optional[int]:
        """Число гостей словами: «Один» в ответ на «сколько человек», «нас двое», «человек один», «вдвоём»."""
        toks = norm.split()
        for i, w in enumerate(toks):
            n = WORD_NUMS.get(w)
            if n is None:
                continue
            prev, nxt = toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else ""
            if (expecting == "guests" and len(toks) <= 4) or prev in ("нас", "будет", "я", "человек", "гостей", "гость") \
                    or nxt in ("человек", "человека", "гостя", "гостей") or w.startswith(("вдво", "втро", "вчет")):
                return n
        return None

    async def _cancel_core(self, chat_id: int, sess: Session, confirmed: bool) -> str:
        """Отмена брони делается ТОЛЬКО здесь, по базе: показать бронь и спросить -> гость подтвердил -> cancel_active -> проверить,
        что бронь действительно исчезла. Слова модели «Отменяю бронь» сами по себе ничего не отменяют и ничему не верим."""
        now = self.now()
        try:
            a = await asyncio.to_thread(self.db.active_booking, chat_id, now)
        except Exception:
            log.exception("chat=%s: не удалось получить бронь из базы", chat_id)
            return DB_FAIL_TEXT
        if not a:
            sess.cancel_armed = False
            return "Активной брони у вас нет — отменять нечего."
        if not confirmed:
            sess.cancel_armed = True
            return f"У вас есть бронь: {booking_line(a.date, a.time, a.guests, a.name)}. Отменить её?"
        try:
            old = await asyncio.to_thread(self.db.cancel_active, chat_id, now)
            sess.cancel_armed, sess.cancelled = False, old or a
            still = await asyncio.to_thread(self.db.active_booking, chat_id, now)
        except Exception:
            log.exception("chat=%s: не удалось отменить бронь в базе", chat_id)
            return "Не получилось отменить бронь. Обратитесь, пожалуйста, к администратору ресторана."
        if still and (not old or (still.date, still.time) == (a.date, a.time)):
            log.error("chat=%s: cancel_active не снял бронь %s %s (old=%r)", chat_id, a.date, a.time, old)
            return "Не получилось отменить бронь. Обратитесь, пожалуйста, к администратору ресторана."
        return f"Бронь отменена: {booking_line(a.date, a.time, a.guests, a.name)}."

    @staticmethod
    def _norm(text: str) -> str:
        """Нижний регистр, «ё» -> «е», без знаков препинания."""
        return " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("ё", "е")).split())

    @staticmethod
    def _stem(w: str) -> str:
        n = len(w)
        return w if n <= 4 else w[:4] if n <= 6 else w[:5] if n <= 8 else w[:n - 4]     # «мидии» -> «мид», «курицей» -> «кури»

    async def _code_answer(self, chat_id: int, sess: Session, clean: str, norm: str) -> Optional[TurnResult]:
        """Ответы кода без модели. Любая ошибка здесь не должна ломать бота: пишем traceback в лог и отдаём вопрос модели."""
        try:
            routed = tg.strip_greeting(clean)                            # «Добрый день. Покажите меню» — разбираем «Покажите меню»
            return await self._code_answer_inner(chat_id, sess, clean, self._norm(routed) if routed != clean else norm, routed)
        except Exception:
            log.exception("chat=%s: код-ответ на %r упал, вопрос уходит модели", chat_id, clean)
            return None

    async def _code_answer_inner(self, chat_id: int, sess: Session, clean: str, norm: str, routed: Optional[str] = None) -> Optional[TurnResult]:
        """Вопросы, на которые отвечает код без модели: «есть ли у меня бронь», «есть ли блюдо X», цена, острое, совет."""
        last_bot = re.sub(r"<[^>]+>", "", " ".join(sess.prev_msgs)).lower()
        asked_cancel = "отменить" in last_bot and "?" in last_bot            # бот только что спросил «Отменить её?»
        if CANCEL_REQ.search(norm) and not re.search(r"заброни|нов[уы]ю|хочу брон", norm) or (
                sess.cancel_armed and asked_cancel and len(norm.split()) <= 3 and AFFIRM.search(f" {norm} ") and not re.match(r"нет|неа", norm)):
            confirmed = sess.cancel_armed and asked_cancel and bool(AFFIRM.search(f" {norm} "))
            text = await self._cancel_core(chat_id, sess, confirmed)
            return self._direct(sess, clean, text, "код: отмена брони", close="Отменить её?" not in text)
        if (CHECK_BOOKING.search(norm) and "отмен" not in norm) or (CHECK_BOOKING2.search(norm) and not NOT_CHECK.search(norm)):
            try:
                a = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
            except Exception:
                log.exception("chat=%s: не удалось проверить бронь в базе", chat_id)
                return self._direct(sess, clean, DB_FAIL_TEXT, "код: проверка брони", close=False)
            sess.cancel_armed = a is not None
            if a:
                text = f"У вас есть бронь: {booking_line(a.date, a.time, a.guests, a.name)}. Отменить её?"
            else:
                text = "Активной брони у вас сейчас нет. Если хотите забронировать столик — напишите дату, время, количество гостей и имя."
            return self._direct(sess, clean, text, "код: проверка брони", close=a is None)      # «Отменить её?» — «Да» разберёт модель (cancel_armed уже включён)
        if self._expecting(sess):
            return None
        words = norm.split()
        if words and len(words) <= 3 and words[0] in ("нет", "неа") and set(words[1:]) <= {"спасибо", "не", "надо", "нужно", "пока", "уже"}:
            last_bot = re.sub(r"<[^>]+>", "", " ".join(sess.prev_msgs)).lower()          # одинокое «Нет» — не жалоба, а ответ на вопрос бота
            text = ("Хорошо, бронь остаётся без изменений. Могу чем-то ещё помочь?" if sess.cancel_armed or "отмен" in last_bot
                    else "Хорошо. Если понадоблюсь — обращайтесь!")
            sess.reset()                                                 # отказ: собранные данные, «cancel_armed» и контекст сбрасываем
            res = TurnResult([Message(text)], source="код: отказ")
            sess.prev_msgs, sess.prev_raws, sess.prev_user = [text], set(), clean.lower()
            self.stats[res.source] += 1
            return res
        if words and len(words) <= 2 and words[0] in BARE_YES and not sess.cancel_armed:   # «Отменить её?» — «Да» разбирает модель
            last_bot = re.sub(r"<[^>]+>", "", " ".join(sess.prev_msgs)).lower()          # что бот реально отправил в прошлый ход
            if OFFER_BOOK.search(last_bot) and "отмен" not in last_bot:
                sess.booking_ok = True                                   # гость принял предложение оформить бронь
                return self._direct(sess, clean, BOOK_ASK, "код: бронь", close=False)
            if "?" not in last_bot and "отмен" not in last_bot:          # на «Да» не было вопроса — не гадаем
                return self._direct(sess, clean, HELP_ASK, "код: меню")
        if UNSUPPORTED.search(norm) and not BOOKING_WORDS.search(norm):
            return self._direct(sess, clean, UNSUPPORTED_TEXT, "код: не умеет")
        m_cap = PRICE_CAP.search(norm)
        if m_cap and PRICE_CAP_CTX.search(norm) and not re.search(r"человек|гост|брон|столик|\d\s*:\s*\d|час", norm) and len(norm.split()) <= 10:
            cap = int(m_cap.group(1)) * (1000 if (m_cap.group(2) or "").startswith("тыс") else 1)
            items = sorted((i for i in _regular(self._menu()) if i.price is not None and i.price <= cap), key=lambda i: (i.price, i.name))
            if items:                                                    # «блюда дешевле 500 рублей» — по ценам из menu.json
                msgs = _list_message(f"Блюда до {fmt_price(cap)}", items[:12])
                res = self._direct(sess, clean, msgs[0], "код: цена", html=True, hist=f"Показал блюда до {cap} ₽: " + ", ".join(i.name for i in items[:12]))
                res.messages += [Message(t, True) for t in msgs[1:]]
                return res
            return self._direct(sess, clean, f"Блюд до {fmt_price(cap)} в нашем меню не нашёл.", "код: цена")
        if not tg.output_ok("review", [norm]) and (STAFF.search(norm) or (re.search(r"шеф", norm) and not re.search(r"блюд|меню|совет|рекоменд|фирменн", norm))) \
                and len(norm.split()) <= 10 and not BOOKING_WORDS.search(norm):
            return self._direct(sess, clean, STAFF_TEXT, "код: нет данных")              # про персонал и вакансии у бота данных нет
        if NO_INFO.search(norm) and len(norm.split()) <= 8 and not BOOKING_WORDS.search(norm):
            return self._direct(sess, clean, NO_INFO_TEXT, "код: нет данных")      # нет в данных бота — не выдумываем
        if ALCOHOL_Q.search(norm) and len(norm.split()) <= 10 and not BOOKING_WORDS.search(norm) \
                and not (_dish_hits(routed or clean, self._menu()) and (PRICE_RE.search(norm) or COMPOSE_RE.search(norm) or SPICY_Q_RE.search(norm) or MEAT_Q_RE.search(norm))):
            return self._alcohol_answer(sess, clean, norm)               # алкоголь — по меню, а условия (свой алкоголь, возраст) — у администратора
        if CAPS_WORDS.search(norm) and len(norm.split()) <= 8:          # «Что ты умеешь?» — не повод рекомендовать блюдо
            return self._direct(sess, clean, CAPS_TEXT, "код: меню")
        if self.topic_gate and not (sess.slots or sess.cancel_armed or (not sess.ctx_closed and self._booking_in_progress(sess))):
            named = bool(_dish_hits(routed or clean, self._menu()) or _category_of(_tokens(routed or clean)))
            av = self._avail_topic(norm)                                 # «а у вас есть хинкали» — слово незнакомое, но вопрос про наличие: ищем в меню
            if av and len(av[0]) <= 2 and not any(NONFOOD.search(w) for w in av[0]):
                named = True
            if (sess.menu_ctx.get("names") or sess.menu_ctx.get("lunch_names")) and tg.followup_shape(norm):
                named = True                                             # «а что туда входит?» после показанного блюда — продолжение, а не новая тема
            if not tg.has_domain_evidence(norm, named):                  # «В чём смысл жизни?», «посоветуй рестораны» — не наша тема
                return self._direct(sess, clean, OFFTOPIC_TEXT, "код: не по теме")
        if ADDRESS_Q.search(norm) and not BOOKING_WORDS.search(norm) and len(norm.split()) <= 9:   # адрес и дорога — факты из кода, не пересказ модели
            return self._direct(sess, clean, ADDRESS_TEXT + (TRAVEL_TEXT if TRAVEL_Q.search(norm) else ""), "код: адрес")
        if HOURS_Q.search(norm) and not re.search(r"бизнес|ланч|lunch|обед|брон|столик", norm) and len(norm.split()) <= 9:
            return self._direct(sess, clean, HOURS_TEXT, "код: часы работы")
        if COMPLAINT_STRONG.search(norm):
            return None                                                  # жалоба/отзыв — дело модели и notify_manager
        if self.use_menu_router:                                         # меню, цены, состав, острое, рекомендации — menu_router по menu.json
            msgs = route(routed or clean, self._menu(), ctx=sess.menu_ctx)
            if msgs:
                plain = html_lib.unescape(re.sub(r"<[^>]+>", "", msgs[0])).strip()
                res = self._direct(sess, clean, msgs[0], "код: меню", html=True, hist=plain[:400])
                res.messages += [Message(t, True) for t in msgs[1:]]
                return res
        found = self._avail_topic(norm)
        return self._answer_avail(sess, clean, *found) if found else None

    def _alcohol_answer(self, sess: Session, clean: str, norm: str) -> TurnResult:
        """Вопросы про алкоголь: что из алкогольного есть в меню (по названию и описанию), иначе честно «не нашёл — уточните у администратора»."""
        menu = self._menu()
        if ALCOHOL_BYO.search(norm):
            return self._direct(sess, clean, "Условия, можно ли приносить свой алкоголь, лучше уточнить у администратора ресторана.", "код: алкоголь")
        spec = [w for w in re.findall(r"[а-я]{3,}", norm) if re.match(r"пив|вин|саке|сак[эе]|соджу|виски|водк|шампанск|просекко|коньяк|ликер|сидр|джин|текил|ром$|мартини|коктейл", w)]
        pool = [i for i in _regular(menu) if i.category == "drinks" or ALCOHOL_ITEM.search(menu_norm(i.name))]
        if spec:                                                         # «а пиво есть?», «саке?» — по названию/описанию
            stems = [w[:4] if len(w) > 4 else w for w in spec]
            hits = [i for i in pool if any(st in menu_norm(f"{i.name} {i.description or ''}") for st in stems)]
        else:                                                            # «у вас есть алкоголь?» — всё алкогольное из меню
            hits = [i for i in pool if ALCOHOL_ITEM.search(menu_norm(f"{i.name} {i.description or ''}")) and "безалкогол" not in menu_norm(f"{i.name} {i.description or ''}")]
        if hits:
            msgs = _list_message("Алкогольные напитки" if not spec else "Вот что нашлось", hits)
            res = self._direct(sess, clean, msgs[0], "код: алкоголь", html=True, hist="Показал напитки из меню: " + ", ".join(i.name for i in hits))
            res.messages += [Message(t, True) for t in msgs[1:]]
            return res
        what = "такого напитка" if spec else "алкогольных напитков"
        return self._direct(sess, clean, f"В нашем меню {what} я не нашёл. Уточните, пожалуйста, у администратора ресторана. "
                                         "Могу показать раздел «Напитки» — напишите «покажи напитки».", "код: алкоголь")

    def _direct(self, sess: Session, clean: str, text: str, source: str, close: bool = True, html: bool = False,
                hist: Optional[str] = None) -> TurnResult:
        """Ответ кода: попадает в историю диалога (hist — его текстом без разметки), чтобы модель видела его в следующих репликах."""
        try:
            sess.history.prompt_for_user(clean)
            sess.history.commit(hist or text)
            if close and not (sess.slots or sess.pending_book):          # шла бронь: вопрос про меню/цену не должен стирать собранные дату, время, гостей
                sess.close_dialog()
        except Exception:                                                # история — только подсказка модели, ответ гостю всё равно уходит
            log.exception("не удалось записать ответ кода в историю диалога")
            sess.history.reset()
        res = TurnResult([Message(text, html)], source=source)
        sess.prev_msgs, sess.prev_raws, sess.prev_user = [text], set(), clean.lower()
        self.stats[source] += 1
        return res

    def _all_items(self) -> list:
        return list(self._menu().items)

    def _find(self, words: list[str]) -> dict:
        """Блюда, в названии/описании которых есть ВСЕ слова (начало слова; «рыба», «лапша» раскрываются синонимами)."""
        items, found = self._all_items(), None
        for w in words:
            stems = next((v for k, v in SYNONYMS.items() if w.startswith(k)), (self._stem(w),))
            hits = {}
            for it in items:
                hay = " ".join([it.name, *it.aliases, it.description or ""]).lower().replace("ё", "е")
                if not it.business_lunch and any(re.search(rf"(?<![а-яa-z]){re.escape(st)}", hay) for st in stems):
                    hits[it.name] = it
            found = hits if found is None else {k: v for k, v in found.items() if k in hits}
        return found or {}

    @classmethod
    def _avail_topic(cls, norm: str) -> Optional[tuple[list[str], bool]]:
        """«у вас есть мидии?», «а с рыбой?» -> (['мидии'], True); иначе None (это не вопрос про блюдо).
        Второе значение: можно ли отвечать «не нашёл». «Какой у вас рейтинг?» — не про блюдо, и если в меню ничего нет,
        отвечать «нет такого блюда» нельзя: пусть отвечает модель."""
        if len(norm.split()) > 8 or BOOKING_WORDS.search(norm) or COMPOSE_RE.search(norm):      # «из чего X» — про состав, не про наличие
            return None
        for rx in AVAIL:
            m = rx.search(norm)
            if m:
                topic, miss_ok = m.group(1), rx is not AVAIL[2]
                break
        else:
            return None
        if re.search(r"\d", topic):
            return None
        words = [w for w in re.findall(r"[a-zа-я]{3,}", topic) if w not in AVAIL_STOP]
        if any(w.startswith(AVAIL_SKIP) for w in words):
            return None
        words = [w for w in words if not w.startswith(AVAIL_GENERIC)]
        return (words, miss_ok) if words else None

    def _menu_vocab(self) -> dict:
        """Слова меню (названия, aliases, описания) -> блюда, где они встречаются: для поиска похожих слов («капучинно» ~ «каппучино»)."""
        m = self._menu()
        if getattr(self, "_vocab_cache", (None, None))[0] is m:
            return self._vocab_cache[1]
        vocab: dict = {}
        for it in self._all_items():
            if it.business_lunch:
                continue
            for w in re.findall(r"[a-zа-я]{4,}", menu_norm(" ".join([it.name, *it.aliases, it.description or ""]))):
                vocab.setdefault(w, []).append(it)
        self._vocab_cache = (m, vocab)
        return vocab

    def _similar(self, words: list[str]) -> list:
        vocab, out = self._menu_vocab(), {}
        for w in words:
            for v in difflib.get_close_matches(w, list(vocab), n=3, cutoff=0.75):
                for it in vocab[v]:
                    out[it.name] = it
        return list(out.values())

    def _answer_avail(self, sess: Session, clean: str, words: list[str], miss_ok: bool) -> Optional[TurnResult]:
        found = self._find(words)
        topic = " ".join(words)
        if found:
            items = list(found.values())
            lines = [f"• «{it.name}» — {fmt_price(it.price)}" for it in items[:AVAIL_MAX]]
            more = f"\nИ ещё: {len(items) - AVAIL_MAX}." if len(items) > AVAIL_MAX else ""
            text = "Да, в меню есть:\n" + "\n".join(lines) + more
        elif miss_ok:
            sim = self._similar(words)                                   # похоже на что-то из меню (опечатка, другая форма слова)
            if sim:
                text = (f"Точно «{topic}» в меню не нашёл. Возможно, вы имели в виду:\n"
                        + "\n".join(f"• «{it.name}» — {fmt_price(it.price)}" for it in sim[:AVAIL_MAX]))
            else:
                dh = next(((rx2, lab) for rx1, rx2, lab in DISH_HINTS if rx1.search(menu_norm(topic))), None)
                alike = [i for i in _regular(self._menu()) if dh and dh[0].search(menu_norm(f"{i.name} {i.description or ''}"))][:AVAIL_MAX]
                if alike:                                                # «хинкали» — по формату похожи гёдза
                    text = (f"«{topic.capitalize()}» в нашем меню не нашёл. Похожее ({dh[1]}):\n"
                            + "\n".join(f"• «{it.name}» — {fmt_price(it.price)}" for it in alike))
                    return self._direct(sess, clean, text, "код: меню")
                hint = next(((c, lab) for c, lab, rx in CAT_HINTS if rx.search(menu_norm(topic))), None)
                items = [i for i in _regular(self._menu()) if hint and i.category == hint[0]][:AVAIL_MAX]
                if items:                                                # «капучино» — напиток: показываем раздел, где похожее
                    msgs = _list_message(hint[1], items)
                    res = self._direct(sess, clean, f"«{topic.capitalize()}» в нашем меню не нашёл. Из раздела «{hint[1]}» могу предложить:", "код: меню",
                                       hist=f"«{topic}» в меню нет; показал раздел {hint[1]}: " + ", ".join(i.name for i in items))
                    res.messages += [Message(t, True) for t in msgs]
                    return res
                text = f"Блюд и напитков со словом «{topic}» в нашем меню не нашёл. Могу подсказать, что есть в меню."
        else:
            return None
        log.info("вопрос про блюдо %r решил код: найдено %d", topic, len(found or {}))
        return self._direct(sess, clean, text, "код: меню")

    @staticmethod
    def _slots_complete(sess: Session) -> bool:
        return all(sess.slots.get(k) for k in ("date", "time", "guests", "name"))

    async def _drive_booking(self, chat_id: int, sess: Session, clean: str) -> TurnResult:
        """Все четыре поля названы гостем (их нашёл код): дату, время, лимит гостей, активную бронь и свободный стол проверяет сам код
        по базе, затем гость подтверждает. Модель текстом «занят» / «записал» больше ничего не решает."""
        s, now = sess.slots, self.now()
        sess.booking_ok = True
        p = {"DATE": str(s["date"]), "TIME": str(s["time"]), "GUESTS": str(s["guests"]), "NAME": str(s["name"])}
        err = validate_booking_date(p["DATE"], now.date())
        if err:
            sess.slots.pop("date", None)
            return self._direct(sess, clean, err, "код: бронь", close=False)
        result, direct = await self._run_tool(chat_id, sess, Reply(raw="", action="call_free_tables_agent", params=p))
        if direct:
            if direct != GUEST_LIMIT_TEXT:
                sess.slots.pop("time", None)                             # время неверно — переспросим (число гостей _run_tool уже убрал)
            return self._direct(sess, clean, direct, "код: бронь", close=False)
        if str(result).startswith("active_booking"):
            a = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            text = f"У вас есть бронь: {booking_line(a.date, a.time, a.guests, a.name)}. Отменить её?" if a else \
                "У вас уже есть активная бронь. Отменить её?"
            return self._direct(sess, clean, text, "код: бронь", close=False)
        if result != "available":
            w = "человека" if int(p["GUESTS"]) % 10 in (2, 3, 4) and int(p["GUESTS"]) % 100 not in (12, 13, 14) else "человек"
            sess.slots.pop("time", None)
            return self._direct(sess, clean, f"К сожалению, {p['DATE']} в {p['TIME']} столик на {p['GUESTS']} {w} занят. "
                                             "Напишите, пожалуйста, другое время, и я проверю снова.", "код: бронь", close=False)
        out = TurnResult()
        self._ask_confirm(sess, p, out)
        return self._direct(sess, clean, out.messages[0].text, "код: бронь", close=False)

    def _booking_in_progress(self, sess: Session) -> bool:
        """Шла ли бронь до этой реплики: код уже что-то собрал, или бот / гость в последних репликах говорили о брони.
        Вопрос бота может быть сразу про всё («Напишите дату, время, количество человек и имя») — _expecting его не узнает."""
        if sess.booking_ok or sess.slots or sess.pending_book or self._expecting(sess):
            return True
        last_bot = sess.history.done[-1].split("-> Bot:")[-1] if sess.history.done else ""
        prev_user = sess.user_texts[-1] if sess.user_texts else ""      # реплика гостя ДО текущей (текущая ещё не добавлена)
        return bool(BOOKING_WORDS.search(f"{last_bot} {prev_user}".lower()))

    def _booking_wanted(self, sess: Session) -> bool:
        """Гость сам говорит о брони: слова «бронь/столик/записаться» или хотя бы две детали (дата, время, гости, имя)."""
        last = sess.user_texts[-1] if sess.user_texts else ""
        said = " ".join(sess.user_texts).lower()
        
        # Гость только что сказал, что не хочет бронировать (старые реплики не считаем: иначе после «уже не надо»
        # о чём-то другом бронь была бы заблокирована до конца разговора)
        if ABORT_BOOKING.search(" ".join(re.sub(r"[^\w\s]", " ", last.lower().replace("ё", "е")).split())):
            sess.booking_ok = False 
            return False
            
        return (sess.booking_ok or bool(BOOKING_WORDS.search(said)) or len(sess.slots) >= 2
                or ("date" in sess.last_slots and "?" not in last and len(last.split()) <= 3))

    def _ground(self, sess: Session, reply: Reply) -> list[str]:
        """Подставляет в команду модели дату/время/гостей/имя из слов гостя (их нашёл код).
        Возвращает поля, которых гость ещё не назвал, — их бот спросит, а не придумает."""
        p, s, said = reply.params, sess.slots, " ".join(sess.user_texts).lower()
        if reply.action == "call_date_agent":
            if s.get("date"):
                p["EXPRESSION"] = s["date"]
                return []
            return [] if p["EXPRESSION"].lower()[:4] in said else ["DATE"]
        tool_dates = re.findall(r"Tool: RESULT: (\d\d\.\d\d\.\d{4})", " ".join(sess.history.done))
        model_hour = re.match(r"(\d{1,2})", p.get("TIME", "")) or None
        vals = {"DATE": s.get("date") or (p.get("DATE") if p.get("DATE") in tool_dates else None),
                "TIME": s.get("time") or (p.get("TIME") if model_hour and TIME_SAID.search(said) and re.search(
                    rf"(?<![\d.:]){int(model_hour.group(1))}(?![\d.]|\s*(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек|человек|чел|гост|персон))", said) else None),
                "GUESTS": s.get("guests") or (p.get("GUESTS") if p.get("GUESTS") and re.search(
                    rf"(?<![\d:.]){re.escape(p['GUESTS'])}(?![\d:.]|\s*(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек|час|ч\b|руб|₽))", said) else None),
                "NAME": s.get("name") or (self._name_from_user(sess, p["NAME"]) if p.get("NAME") else None)}
        if vals["NAME"] and vals["NAME"].lower()[:4] not in said:
            vals["NAME"] = None
        for k, v in vals.items():
            if v and k in p and str(v) != p[k]:
                reply.raw = reply.raw.replace(f"{k}: {p[k]}", f"{k}: {v}")
                p[k] = str(v)
        return [k for k in NEED[reply.action] if not vals[k]]

    @staticmethod
    def _ack_ok(sess: Session, text: str) -> bool:
        """«Понял: дата — …, время — …, гостей — …» модели совпадает с тем, что сказал гость (и что вернул календарь)?"""
        tool_dates = re.findall(r"Tool: RESULT: (\d\d\.\d\d\.\d{4})", " ".join(sess.history.done + [sess.history._open or ""]))
        for lab, val in re.findall(r"(дата|время|гостей|имя) — ([^,.?]+(?:\.\d\d\.\d{4})?)", text):
            key = {"дата": "date", "время": "time", "гостей": "guests", "имя": "name"}[lab]
            mine = str(sess.slots.get(key, ""))
            if val.strip() != mine and not (key == "date" and val.strip() in tool_dates):
                return False
        said = {k for k in ("date", "time", "guests", "name") if k in sess.slots}
        named = {{"дата": "date", "время": "time", "гостей": "guests", "имя": "name"}[lab]
                 for lab in re.findall(r"(дата|время|гостей|имя) — ", text)}
        return said <= named                                             # модель не «забыла» то, что гость уже сказал

    @staticmethod
    def _ask_missing(sess: Session, missing: list[str]) -> Optional[str]:
        """«Записал: дата — 21.10.2026, время — 19:00. Подскажите, пожалуйста, сколько человек будет и на чьё имя оформить бронь?»"""
        if not missing:
            return None
        known = [f"{lab} — {sess.slots[key]}" for k, key, _, lab in FIELDS if key in sess.slots]
        asks = [q for k, _, q, _ in FIELDS if k in missing]
        ask = ", ".join(asks[:-1]) + " и " + asks[-1] if len(asks) > 1 else asks[0]
        return (f"Записал: {', '.join(known)}. " if known else "") + f"Подскажите, пожалуйста, {ask}?"

    @staticmethod
    def _expecting(sess: Session) -> Optional[str]:
        """Что бот только что спросил (одно поле) — помогает понять короткий ответ «19» или «Рита»."""
        if not sess.history.done:
            return None
        q = sess.history.done[-1].split("-> Bot:")[-1].lower()
        asked = [key for key, ws in (("date", ("какую дату", "какой день")), ("time", ("какое время", "какое точное время", "во сколько")),
                                     ("guests", ("сколько человек", "сколько вас", "сколько гостей")),
                                     ("name", ("чьё имя", "чье имя", "какое имя", "как вас зовут"))) if any(w in q for w in ws)]
        return asked[0] if len(asked) == 1 else None

    @staticmethod
    def _name_from_user(sess: Session, name: str) -> str:
        """Модель написала «Марат», а гость — «Маргарита»: берём слово гостя с тем же началом."""
        said = " ".join(sess.user_texts)
        if name.lower()[:4] in said.lower():
            return name
        for w in reversed(re.findall(r"[А-ЯЁA-Z][а-яёa-z]{2,}", said)):
            if w[:2].lower() == name[:2].lower() and w.lower() not in YES:
                return w
        return name

    def _dish_on_menu(self, user_text: str) -> Optional[str]:
        """Модель сочла вопрос «не по теме», а гость назвал блюдо из меню — отвечаем по menu.json."""
        for w in re.findall(r"[а-яёa-z]{4,}", user_text.lower()):
            items = find_items(w, self._menu())
            if len(items) == 1:
                it = items[0]
                return f"Да, у нас есть «{it.name}» — {fmt_price(it.price)}" + (f": {it.description.lower()}." if it.description else ".")
        return None

    @staticmethod
    def _lunch_day(user_text: str, text: str) -> str:
        """«А во вторник есть бизнес-ланч?» — день недели проверяет код (модель путает дни)."""
        t = user_text.lower()
        if "ланч" not in t:
            return text
        asked = [k for k in WEEKDAYS if k in t]
        if not asked or re.search(r"\d", t):
            return text
        low = text.lower()
        said_no = not low.startswith("да") and bool(re.search(r"(?<![а-яё])нет(?![а-яё])", low))
        other_day = any(k in text.lower() and k not in asked for k in ("вторник", "сред", "четверг"))
        if said_no != bool(WEEKDAYS[asked[0]]) and not other_day:
            return text                                                  # модель ответила верно
        return (f"Да, бизнес-ланчи у нас {LUNCH_SCHED}." if WEEKDAYS[asked[0]] else
                f"Нет, бизнес-ланчи у нас только по будням: {LUNCH_SCHED}.")

    async def _check_facts(self, chat_id: int, sess: Session, res: TurnResult) -> None:
        """После записи брони или показа брони модель пересказывает её своими словами — сверяем дату, время,
        гостей и имя с базой; при расхождении говорим данные из базы."""
        kinds = [str(st.get("result") or "") for st in res.steps]
        if not any(k in ("booked", "cancelled") or k.startswith("active_booking") for k in kinds):
            return
        gone = "cancelled" in kinds and "booked" not in kinds
        a = sess.cancelled if gone else await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
        if not a:
            return
        if "booked" in kinds and res.messages and not res.messages[-1].html:
            # итог брони пишет код: модель путала формат и искажала имя («Имя:ана» вместо «Анатолий»)
            line = booking_line(a.date, a.time, a.guests, a.name)
            final = f"Готово, бронь подтверждена: {line}. Ждём вас в YUQI!"
            if res.messages[-1].text != final:
                log.info("chat=%s: итог брони написан кодом, модель сказала: %r", chat_id, res.messages[-1].text)
            res.messages[-1] = Message(final)
            res.source = "код: проверка брони"
            return
        for i, m in enumerate(res.messages):
            t = m.text
            if m.html or not re.search(r"\d\d\.\d\d\.\d{4}|[Ии]мя", t):
                continue
            bad = (any(d != a.date for d in re.findall(r"\d\d\.\d\d\.\d{4}", t))
                   or any(x != a.time for x in re.findall(r"(?<!\d)\d\d:\d\d", t))
                   or any(int(g) != a.guests for g in re.findall(r"(\d+) (?:человек|гост)|человек: (\d+)", t) for g in g if g)
                   or any(n.lower() != a.name.lower() for n in re.findall(r"имя\s*[:—-]?\s*([а-яёa-z\-]+)", t, re.I)))
            lead = re.match(r"([А-ЯЁ][а-яё]+), ", t)
            if lead and lead.group(1) != a.name and lead.group(1).lower() not in OK_LEAD:
                t = t[lead.end():]
                t = t[0].upper() + t[1:]                                 # «Влад, бронь подтверждена» -> имя гостя модель не выдумывает
            if bad:
                line = booking_line(a.date, a.time, a.guests, a.name)
                t = (f"Готово, бронь подтверждена: {line}. Ждём вас в YUQI!" if "booked" in kinds else
                     f"Бронь отменена: {line}." if gone else
                     f"У вас есть бронь: {line}." + (" Отменить её?" if "отмен" in m.text.lower() else ""))
                log.warning("chat=%s: модель исказила данные брони: %r", chat_id, m.text)
                res.source = "код: проверка брони"
            res.messages[i] = Message(t)

    async def _check_claims(self, chat_id: int, res: TurnResult) -> None:
        """Любые слова модели о брони сверяются с фактами: «оформлена» — только если база ответила booked,
        «отменена» — только если cancelled, «у вас есть бронь» — только если она есть в базе и с теми же датой/временем."""
        results = " ".join(str(st.get("result") or "") for st in res.steps)
        for i, m in enumerate(res.messages):
            t = m.text.lower()[:1500]
            if m.html or not (CLAIM_BOOKED.search(t) or CLAIM_CANCELLED.search(t) or CLAIM_EXISTS.search(t)):
                continue
            if CLAIM_BOOKED.search(t) and "booked" in results.split():
                continue
            if CLAIM_CANCELLED.search(t) and "cancelled" in results:
                continue
            a = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
            if CLAIM_EXISTS.search(t) and not CLAIM_BOOKED.search(t) and not CLAIM_CANCELLED.search(t) and a \
                    and a.date in m.text and a.time in m.text:
                continue                                                 # модель верно пересказала бронь из базы
            log.warning("chat=%s: слова модели о брони не совпали с базой: %r", chat_id, m.text)
            if CLAIM_CANCELLED.search(t) and "cancelled" not in results:
                res.messages[i] = Message((f"Бронь пока не отменена — она активна: {booking_line(a.date, a.time, a.guests, a.name)}. "
                                           "Чтобы отменить, напишите «отмени бронь».") if a else "Активной брони у вас нет — отменять нечего.")
            else:
                res.messages[i] = Message(f"Сверился с базой: у вас есть бронь — {booking_line(a.date, a.time, a.guests, a.name)}." if a else
                                          "Сверился с базой: активной брони у вас сейчас нет.")
            res.source = "код: проверка брони"

    # ------------------------------------------------------------ проверка формы ответа
    def _problem(self, gen: GenResult, reply: Reply) -> Optional[str]:
        if gen.stop == "max_tokens":
            return "ответ оборван (max_tokens)"
        if not reply.raw.strip():
            return "пустой ответ"
        if reply.has_action:
            errs = validate_action(reply)
            if errs:
                return "некорректное действие: " + "; ".join(errs)
        elif not reply.text:
            return "нет ни текста, ни действия"
        if self.min_logprob is not None and gen.mean_logprob < self.min_logprob:
            return f"низкая уверенность: {gen.mean_logprob:.2f}"
        return None

    # ------------------------------------------------------------ инструменты с результатом
    async def _run_tool(self, chat_id: int, sess: Session, reply: Reply) -> tuple[Optional[str], Optional[str]]:
        """-> (результат для модели, None) или (None, текст для гостя вместо продолжения цепочки)."""
        a, p, now = reply.action, reply.params, self.now()

        if a == "call_date_agent":
            d = resolve_date(p["EXPRESSION"], now.date())
            if d is None:
                return None, BAD_DATE_TEXT
            err = validate_booking_date(d, now.date())
            return (None, err) if err else (d, None)

        if a == "call_booking_time_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem is None:
                return "time_valid", None
            sess.slots.pop("time", None)                                 # время не подошло — спросим другое; текст причины пишет код, а не модель
            return None, problem.text

        if a in ("call_free_tables_agent", "book_table") and int(p["GUESTS"]) > MAX_GUESTS:
            sess.slots.pop("guests", None)
            return None, GUEST_LIMIT_TEXT                                # не «столик занят», а честная причина
        if a == "call_free_tables_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem:
                sess.slots.pop("time", None)
                return None, problem.text
            old = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            if old:                                                      # у гостя уже есть бронь
                sess.cancel_armed = True
                return old.as_exists_result(), None
            ok = await asyncio.to_thread(self.db.has_free_table, int(p["GUESTS"]), p["DATE"], p["TIME"])
            return ("available" if ok else "unavailable"), None

        if a == "book_table":                                            # база заново проверяет время, гостей, занятость
            r = await asyncio.to_thread(self.db.book_table, int(p["GUESTS"]), p["DATE"], p["TIME"], p["NAME"], chat_id, now)
            if r.status in ("booked", "table_busy"):
                return r.status, None
            if r.status == "active_booking_exists":
                sess.cancel_armed = True
                return r.active.as_exists_result(), None
            return None, r.reason

        if a == "check_active_booking":
            old = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            sess.cancel_armed = old is not None
            return (old.as_result() if old else "no_active_booking"), None

        if a == "cancel_booking":
            if not sess.cancel_armed:                                    # модель не показывала бронь гостю — не отменяем
                log.warning("chat=%s: cancel_booking без показанной брони", chat_id)
                return None, "Сначала давайте проверим вашу бронь. Напишите, пожалуйста, что хотите отменить бронь."
            old = await asyncio.to_thread(self.db.cancel_active, chat_id, now)
            sess.cancel_armed, sess.cancelled = False, old
            return ("cancelled" if old else "no_active_booking"), None

        raise ValueError(a)

    # ------------------------------------------------------------ конечные действия
    def _run_terminal(self, chat_id: int, user_name: Optional[str], user_text: str, reply: Reply, out: TurnResult) -> None:
        a, p, m = reply.action, reply.params, self._menu()
        if a == "show_menu":
            out.messages += [Message(t, True) for t in tool_show_menu(p["CATEGORY"], m)]
        elif a == "show_business_lunch":
            out.messages += [Message(t, True) for t in tool_show_business_lunch(m)]
        elif a == "call_menu_filter_agent":
            out.messages += [Message(t, True) for t in filter_messages(p["FILTER"], m)]       # + проверка остроты/мяса по описанию и составу
        elif a == "call_price_agent":
            out.messages.append(Message(tool_price(p["ITEM"], m), True))
        elif a == "notify_manager":                                      # жалоба: гость уже получил ответ модели
            try:
                self.db.save_review(chat_id, user_name, user_text, True, self.now())
            except Exception:
                log.exception("не удалось сохранить отзыв")
            out.notices.append(ManagerNotice(chat_id, user_name, user_text))
