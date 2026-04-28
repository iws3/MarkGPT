# solution to exercise three of strings
# simple cipher
message = "hello"
cipher_text = chr(ord(message[0])+1) + \
                  chr(ord(message[1])+1) + \
                    chr(ord(message[2])+1) + \
                        chr(ord(message[3])+1) + \
                            chr(ord(message[4])+1)
print(f"Original: {message}")
print(f"Encoded: {cipher_text}")