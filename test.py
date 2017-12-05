import asyncio

async def test(n):
	print('testL ', n)

tasks = []
for n in range(3):
	tasks.append(test(n))

print('add to loop')
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(*tasks))
loop.close()
