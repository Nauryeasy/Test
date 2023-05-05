from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.types import ContentType, InputFile
from model import *
import os


global_storage = {}

bot = Bot(TOKEN, parse_mode=types.ParseMode.HTML)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands='start')
async def start(message: types.Message):
    await Form.sticker_not_got.set()
    print('start')
    await message.answer('''
Привет! Я бот, который генерирует стикеры в стиле того, который вы попросите.\n
Для того, чтобы сгенерировать стикер отправте мне пример.\n
Для того, чтобы узнать обо мне поподробнее напишите /help
    '''
    )


@dp.message_handler(content_types=ContentType.STICKER, state=Form.sticker_not_got)
async def say(message: types.Message):
    print('sticker is got')
    try:
        emoji = message.sticker.emoji
        sticker_path = 'style/' + str(message.from_user.id) + '.webp'
        await message.sticker.download(sticker_path)
        global_storage[message.from_user.id] = {'emoji': emoji, 'sticker_path': sticker_path}
        await Form.sticker_got.set()
        await message.answer('Я успешно получил ваш стикер!\nТеперь отправте мне эмодзи, для которого мне нужно сделать стикер')
    except:
        await message.answer('Что-то пошло не по плану :(\nДавайте попробуем еще раз, попробуйте отправить снова!')


@dp.message_handler(state=Form.sticker_got)
async def give_sticker(message: types.Message, state: FSMContext):
    target = message.text
    sticker = InputFile(make_sticker(
        global_storage[message.from_user.id]['sticker_path'],
        global_storage[message.from_user.id]['emoji'],
        target
    ))

    await message.answer_photo(photo=sticker)
    await Form.sticker_not_got.set()
    global_storage.pop(message.from_user.id)
    os.remove('style/' + str(message.from_user.id) + '.webp')


@dp.message_handler(commands='secret', state=Form.sticker_not_got)
async def sticker_pack(message: types.Message):
    await message.answer(
        '''
WARNING! Функция работает как говно, ибо эмодзи какого-то фига тоже говно. \n
Фор эксампле у некоторых эмодзи длинна в четыре символа... Будте осторожны -_-
        \n
        sticker:
        '''
    )
    await Form.make_sticker_pack.set()


@dp.message_handler(state=Form.make_sticker_pack, content_types=ContentType.STICKER)
async def get_info_for_sticker_pack(message: types.Message):
    emoji = message.sticker.emoji
    sticker_path = 'style/' + str(message.from_user.id) + '.webp'
    await message.sticker.download(sticker_path)
    global_storage[message.from_user.id] = {'emoji': emoji, 'sticker_path': sticker_path}


@dp.message_handler(state=Form.make_sticker_pack, content_types=ContentType.TEXT)
async def give_sticker(message: types.Message):
    emojis = message.text
    stickers = []
    for target in emojis:
        print(target)
        sticker = InputFile(make_sticker(
            global_storage[message.from_user.id]['sticker_path'],
            global_storage[message.from_user.id]['emoji'],
            target
        ))
        await message.answer_photo(photo=sticker)

    await Form.sticker_not_got.set()


if __name__ == '__main__':
    executor.start_polling(dp)
