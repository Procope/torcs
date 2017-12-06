import asyncio
from random import randint
from time import sleep

async def test(n):
	s = randint(1,10)
	print('start', n, s)
	await sleep(s)
	print('end', n)

tasks = []
for n in range(3):
	tasks.append(asyncio.ensure_future(test(n)))

print('add to loop')
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(*tasks))
loop.close()
